import os
import gc
import re
import torch
import numpy as np
import cv2
import traceback
from threading import Lock
from PIL import Image

# Diffusers 핵심 라이브러리
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline, 
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel, 
    AutoencoderKL, 
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler
)

# [수정] workers.py와의 호환성을 위해 변수명을 _global_load_lock으로 복구합니다.
# (기능은 여전히 CPU RAM 로딩 구간만 잠그는 역할을 합니다)
_global_load_lock = Lock()

class SDEngine:
    def __init__(self, config, device_id=0):
        self.cfg = config
        self.device = torch.device(f"cuda:{device_id}")
        self.pipe = None
        self.is_sdxl = False 
        
    def _cleanup_memory(self):
        """메모리 정리"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def load(self, checkpoint_name=None):
        """
        [Simple & Fast Strategy]
        1. Lock: 모델을 CPU로 읽어들이는 구간만 Lock (짧게)
        2. Move to GPU: Lock 해제 후 각자 GPU로 이동 (병렬)
        """
        # [수정] 변수명 복구 (_ram_load_lock -> _global_load_lock)
        global _global_load_lock
        
        if checkpoint_name is None:
            checkpoint_name = self.cfg['files']['checkpoint_file']

        paths = self.cfg['paths']
        ckpt_path = os.path.join(paths['checkpoint'], checkpoint_name)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        file_size_gb = os.path.getsize(ckpt_path) / (1024**3)
        self.is_sdxl = "xl" in checkpoint_name.lower() or file_size_gb > 4.5
        
        print(f"[SDEngine] Prepare loading on {self.device}...")

        # 1. 기존 파이프라인 정리 (Lock 없이 수행)
        self._cleanup_memory()

        # 임시 변수
        temp_vae = None
        temp_cnet = None
        temp_pipe = None

        # =========================================================
        # [Phase 1] CPU RAM 로딩 구간 (Lock 사용 - RAM OOM 방지)
        # =========================================================
        with _global_load_lock:
            try:
                print(f"[SDEngine] Loading to CPU RAM (Locked)...")

                # 공통 옵션: 무조건 CPU에 다 싣는다. (Meta Error / Device Map Error 원천 봉쇄)
                cpu_load_opts = {
                    "torch_dtype": torch.float32, 
                    "low_cpu_mem_usage": False,   # 핵심: Accelerate 개입 차단
                    "use_safetensors": True,
                    "device_map": None            # 핵심: 자동 분산 차단
                }

                # 1. ControlNet 로드
                cnet_dir = paths['controlnet']
                if self.is_sdxl:
                    cnet_file = "controlnet-canny-sdxl-1.0.safetensors" 
                    cnet_config = "diffusers/controlnet-canny-sdxl-1.0"
                else:
                    cnet_file = self.cfg['files']['controlnet_tile']
                    cnet_config = "lllyasviel/control_v11f1e_sd15_tile"

                cnet_full_path = os.path.join(cnet_dir, cnet_file)
                if os.path.exists(cnet_full_path):
                    temp_cnet = ControlNetModel.from_single_file(cnet_full_path, config=cnet_config, **cpu_load_opts)

                # 2. VAE 로드
                vae_path = os.path.join(paths['vae'], self.cfg['files']['vae_file'])
                print(f"   - VAE Loading: {os.path.basename(vae_path)}")
                temp_vae = AutoencoderKL.from_single_file(vae_path, **cpu_load_opts)

                # 3. Pipeline 로드 (UNet)
                if self.is_sdxl:
                    base_config = "stabilityai/stable-diffusion-xl-base-1.0"
                    PipelineClass = StableDiffusionXLControlNetImg2ImgPipeline if temp_cnet else StableDiffusionXLPipeline
                    
                    temp_pipe = StableDiffusionXLPipeline.from_single_file(
                        ckpt_path, config=base_config, vae=temp_vae, **cpu_load_opts
                    )
                else:
                    base_config = "runwayml/stable-diffusion-v1-5"
                    PipelineClass = StableDiffusionControlNetImg2ImgPipeline if temp_cnet else StableDiffusionPipeline
                    
                    temp_pipe = StableDiffusionPipeline.from_single_file(
                        ckpt_path, config=base_config, vae=temp_vae, safety_checker=None, **cpu_load_opts
                    )
                
                # 4. 재조립 (CPU 상태)
                init_args = {
                    "vae": temp_pipe.vae,
                    "text_encoder": temp_pipe.text_encoder,
                    "tokenizer": temp_pipe.tokenizer,
                    "unet": temp_pipe.unet,
                    "scheduler": temp_pipe.scheduler,
                }
                
                if temp_cnet:
                    init_args["controlnet"] = temp_cnet
                
                if self.is_sdxl:
                    init_args["text_encoder_2"] = temp_pipe.text_encoder_2
                    init_args["tokenizer_2"] = temp_pipe.tokenizer_2
                    if hasattr(temp_pipe, 'image_processor'):
                        init_args["image_processor"] = temp_pipe.image_processor
                else:
                    init_args["safety_checker"] = None
                    init_args["feature_extractor"] = None
                    if hasattr(temp_pipe, "feature_extractor"):
                        init_args["feature_extractor"] = temp_pipe.feature_extractor

                self.pipe = PipelineClass(**init_args)
                self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

            except Exception as e:
                print(f"[SDEngine] RAM Load Failed: {e}")
                self._cleanup_memory()
                raise e

        # =========================================================
        # [Phase 2] GPU 이동 구간 (Lock 해제됨 - 병렬 처리)
        # =========================================================
        try:
            print(f"[SDEngine] Moving to GPU: {self.device}...")
            
            # 통째로 이동 (데이터가 이미 실존하므로 에러 없음)
            self.pipe.to(self.device)

            # 정밀도 최적화 (Memory Casting)
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.text_encoder.to(dtype=torch.float16)
            
            if self.is_sdxl:
                self.pipe.text_encoder_2.to(dtype=torch.float16)
            
            if temp_cnet:
                self.pipe.controlnet.to(dtype=torch.float16)
                
            # VAE FP32 고정
            self.pipe.vae.to(dtype=torch.float32)

            print(f"[SDEngine] READY on {self.device}")

        except Exception as e:
            print(f"[SDEngine] GPU Move Failed: {e}")
            self._cleanup_memory()
            raise e

    def parse_lora(self, prompt):
        """[안전장치] LoRA 적용"""
        lora_pattern = r"<lora:([^:>]+):?([0-9.]*)?>"
        matches = re.findall(lora_pattern, prompt)
        
        try: self.pipe.unfuse_lora()
        except: pass
        try: self.pipe.unload_lora_weights()
        except: pass
        
        adapters, weights = [], []
        for lora_name, weight_str in matches:
            weight = float(weight_str) if weight_str else 1.0
            full_path = os.path.join(self.cfg['paths']['lora'], f"{lora_name}.safetensors")
            
            if os.path.exists(full_path):
                try:
                    adapter_name = re.sub(r'\W+', '_', lora_name) 
                    self.pipe.load_lora_weights(full_path, adapter_name=adapter_name)
                    adapters.append(adapter_name)
                    weights.append(weight)
                except Exception as e:
                    print(f"[WARN] LoRA load failed: {lora_name} - {e}")
            else:
                print(f"[WARN] LoRA file not found: {full_path}")

        if adapters:
            self.pipe.set_adapters(adapters, adapter_weights=weights)

        clean_prompt = re.sub(lora_pattern, "", prompt)
        clean_prompt = re.sub(r"\[\s*[:\s]*\]", "", clean_prompt)
        return re.sub(r",\s*,", ",", clean_prompt).strip()

    def run(self, image, prompt, neg_prompt, strength, use_cnet, hires_fix=False, callback_on_step=None):
        if self.pipe is None:
            # 파이프라인 없으면 1회 재시도 (즉시 실행)
            try:
                print("[SDEngine] Pipeline lost. Reloading immediately...")
                self.load()
            except:
                raise RuntimeError("Critical: Pipeline reload failed.")

        clean_prompt = self.parse_lora(prompt)
        
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        else:
            image_pil = image

        def internal_callback(pipe, step_index, timestep, callback_kwargs):
            if callback_on_step: callback_on_step(step_index, timestep, callback_kwargs)
            return callback_kwargs

        inference_args = {
            "prompt": clean_prompt,
            "negative_prompt": neg_prompt,
            "image": image_pil,
            "strength": strength,
            "num_inference_steps": 25 if not self.is_sdxl else 30,
            "guidance_scale": 7.5,
            "callback_on_step_end": internal_callback
        }

        if use_cnet and hasattr(self.pipe, "controlnet"):
            inference_args["control_image"] = image_pil
            inference_args["controlnet_conditioning_scale"] = self.cfg['defaults'].get('controlnet_weight', 1.0)
        else:
            if "control_image" in inference_args: del inference_args["control_image"]

        try:
            with torch.inference_mode():
                # [VAE 보호] Autocast는 UNet 연산에만 적용
                with torch.autocast(self.device.type, dtype=torch.float16):
                    pass1_result = self.pipe(**inference_args).images[0]

                    if not hires_fix:
                        return pass1_result

                    print(f"[SDEngine] Hires Fix 활성화 ({'SDXL' if self.is_sdxl else 'SD1.5'})")
                    w, h = image_pil.size
                    scale_factor = 2.0 if not self.is_sdxl else 1.5
                    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                    
                    upscaled_image = pass1_result.resize((new_w, new_h), resample=Image.LANCZOS)
                    
                    inference_args["image"] = upscaled_image
                    inference_args["strength"] = 0.35 
                    inference_args["num_inference_steps"] = 20
                    
                    if "control_image" in inference_args: 
                        inference_args["control_image"] = upscaled_image
                    
                    final_result = self.pipe(**inference_args).images[0]

            return final_result

        except Exception as e:
            print(f"[SDEngine] Run Error: {e}")
            traceback.print_exc()
            return image_pil