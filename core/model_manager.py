import os
import torch
import threading
import gc
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)
from core.config import config_instance as cfg

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, device):
        self.device = device
        self.pipe = None
        self.current_config = {}

    def load_sd_model(self, ckpt_path=None, vae_path=None, controlnet_path=None, clip_skip=1):
        """Stable Diffusion 파이프라인 로드"""
        
        # 기본 경로 설정 (인자가 없으면 config.yaml에서 로드)
        if not ckpt_path:
            ckpt_path = cfg.get_path('checkpoint', 'checkpoint_file')
        if not vae_path:
            vae_path = cfg.get_path('vae', 'vae_file')
        
        # [New] SDXL 판별 (파일명 기반)
        is_sdxl = False
        if ckpt_path:
            fname = os.path.basename(ckpt_path).lower()
            if "xl" in fname or "pony" in fname:
                is_sdxl = True

        # ControlNet 경로 (인자가 없으면 config.yaml에서 로드)
        # [Mod] ControlNet Tile 강제 사용 (설정 추론 에러 방지 및 안정성 확보)
        if not controlnet_path and cfg.get_path('controlnet', 'controlnet_tile'):
            if not is_sdxl:
                controlnet_path = cfg.get_path('controlnet', 'controlnet_tile')
            else:
                print("[ModelManager] SDXL detected. Skipping default SD1.5 ControlNet Tile.")

        # 현재 로드된 설정과 비교
        new_config = {
            'ckpt': ckpt_path, 'vae': vae_path, 
            'cn': controlnet_path, 'clip': clip_skip
        }
        
        if self.pipe is not None and self.current_config == new_config:
            return

        with self._lock:
            if self.pipe: del self.pipe
            self.pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[ModelManager] Loading model: {os.path.basename(ckpt_path)} (ClipSkip: {clip_skip})")
            
            # ControlNet
            controlnet = None
            if controlnet_path:
                cn_path = controlnet_path
                if cn_path and os.path.exists(cn_path) and os.path.isfile(cn_path):
                    try:
                        controlnet = ControlNetModel.from_single_file(cn_path, torch_dtype=torch.float32)
                    except Exception as e:
                        print(f"[ModelManager] Failed to load ControlNet from {cn_path}: {e}")
                else:
                    try:
                        target_path = cn_path
                        if "lllyasviel/control_v11f1e_sd15_tile" in cn_path.replace("\\", "/"):
                            target_path = "lllyasviel/control_v11f1e_sd15_tile"
                        
                        controlnet = ControlNetModel.from_pretrained(target_path, torch_dtype=torch.float32)
                    except Exception as e:
                        print(f"[ModelManager] Failed to load ControlNet (Pretrained) from {cn_path}: {e}")

            # VAE
            vae = None
            if vae_path and os.path.exists(vae_path):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)

            # Args
            load_args = {"torch_dtype": torch.float32, "safety_checker": None, "use_safetensors": True}
            if vae: load_args["vae"] = vae
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            if is_sdxl:
                # [New] SDXL Pipeline Loading
                print(f"[ModelManager] Detected SDXL/Pony model. Using SDXL Pipeline.")
                if controlnet:
                    # SDXL ControlNet Pipeline
                    self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(ckpt_path, controlnet=controlnet, **load_args)
                else:
                    # SDXL Inpaint Pipeline
                    self.pipe = StableDiffusionXLInpaintPipeline.from_single_file(ckpt_path, **load_args)
            else:
                # [Existing] SD 1.5 Pipeline Loading
                try:
                    self.pipe = StableDiffusionInpaintPipeline.from_single_file(ckpt_path, **load_args)
                except OSError as e:
                    print(f"[ModelManager] Warning: Failed to infer config from checkpoint.")
                    print(f"               Error: {e}")
                    print(f"               -> This is a known Diffusers cache issue. Retrying with forced SD 1.5 config...")
                    self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                        ckpt_path, config="runwayml/stable-diffusion-v1-5", **load_args
                    )
                
                if controlnet:
                    self.pipe = StableDiffusionControlNetInpaintPipeline(**self.pipe.components, controlnet=controlnet)

            # Clip Skip 적용
            if not is_sdxl and clip_skip > 1 and hasattr(self.pipe, 'text_encoder'):
                # Diffusers 방식: 레이어 슬라이싱
                # Clip Skip 2 means removing the last 1 layer (using the 2nd to last)
                self.pipe.text_encoder.text_model.encoder.layers = self.pipe.text_encoder.text_model.encoder.layers[:-(clip_skip - 1)]

            # [Fix] OOM 방지를 위한 메모리 최적화 및 Offload 적용
            # 1. 데이터 타입 변환 (fp16)
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.text_encoder.to(dtype=torch.float16)
            if controlnet: self.pipe.controlnet.to(dtype=torch.float16)
            if self.pipe.vae: self.pipe.vae.to(dtype=torch.float32)

            # 2. 메모리 효율화 (xformers / slicing)
            if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    self.pipe.enable_attention_slicing()
            else:
                self.pipe.enable_attention_slicing()

            # 3. VAE Tiling (고해상도 OOM 방지)
            if hasattr(self.pipe.vae, 'enable_tiling'):
                self.pipe.vae.enable_tiling()
            elif hasattr(self.pipe, 'enable_vae_tiling'):
                self.pipe.enable_vae_tiling()

            # 4. CPU Offload (VRAM 절약의 핵심)
            # .to(device) 대신 사용하여 필요할 때만 GPU로 올림
            try:
                self.pipe.enable_model_cpu_offload(device=self.device)
            except Exception as e:
                print(f"[ModelManager] CPU Offload failed ({e}). Fallback to standard .to()")
                self.pipe.to(self.device)

            self.current_config = new_config

    def manage_lora(self, lora_list, action="load"):
        """LoRA 주입 및 해제"""
        if not self.pipe or not lora_list: return
        
        lora_base = cfg.get_path('lora')
        if not lora_base: return

        try:
            if action == "load":
                adapter_names = []
                adapter_weights = []

                for i, (name, scale) in enumerate(lora_list):
                    # 확장자 자동 처리
                    if not name.endswith(('.safetensors', '.ckpt', '.pt')):
                        name += ".safetensors"
                        
                    lora_path = os.path.join(lora_base, name)
                    if os.path.exists(lora_path):
                        # 여러 LoRA를 동시에 로드하기 위해 고유 adapter_name 사용
                        adapter_name = f"lora_{i}"
                        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                        adapter_names.append(adapter_name)
                        adapter_weights.append(scale)
                        print(f"[ModelManager] LoRA Loaded: {name} (Scale: {scale})")
                    else:
                        print(f"[ModelManager] Warning: LoRA not found at {lora_path}")
                
                if adapter_names:
                    self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    self.pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
            
            elif action == "unload":
                try:
                    self.pipe.unfuse_lora()
                    self.pipe.unload_lora_weights()
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception as e:
            print(f"[ModelManager] LoRA Error: {e}")

    def apply_scheduler(self, sampler_name):
        """스케줄러 교체"""
        if self.pipe is None: return
        config = self.pipe.scheduler.config
        
        if "Euler a" in sampler_name:
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        elif "Euler" in sampler_name:
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(config)
        elif "DPM++ 2M" in sampler_name:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
        elif "DDIM" in sampler_name:
            self.pipe.scheduler = DDIMScheduler.from_config(config)