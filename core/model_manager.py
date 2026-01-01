import os
import torch
import threading
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
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
        
        # ControlNet 경로 (인자가 없으면 config.yaml에서 로드)
        # [Fix] ControlNet 경로 강제 설정 제거
        # pipeline.py에서 use_controlnet=False일 때 None을 전달하는데,
        # 여기서 강제로 기본값을 설정하면 의도치 않게 ControlNet 파이프라인이 로드되어
        # control_image 누락 에러(TypeError)가 발생함.
        # if not controlnet_path and cfg.get_path('controlnet', 'controlnet_tile'):
        #      controlnet_path = cfg.get_path('controlnet', 'controlnet_tile')

        # 현재 로드된 설정과 비교
        new_config = {
            'ckpt': ckpt_path, 'vae': vae_path, 
            'cn': controlnet_path, 'clip': clip_skip
        }
        
        if self.pipe is not None and self.current_config == new_config:
            return

        with self._lock:
            if self.pipe: del self.pipe
            torch.cuda.empty_cache()
            print(f"[ModelManager] Loading model: {os.path.basename(ckpt_path)} (ClipSkip: {clip_skip})")
            
            # ControlNet
            controlnet = None
            if controlnet_path:
                cn_path = controlnet_path
                if cn_path and os.path.exists(cn_path) and os.path.isfile(cn_path):
                    try:
                        controlnet = ControlNetModel.from_single_file(cn_path, torch_dtype=torch.float16)
                    except Exception as e:
                        print(f"[ModelManager] Failed to load ControlNet from {cn_path}: {e}")
                else:
                    try:
                        target_path = cn_path
                        if "lllyasviel/control_v11f1e_sd15_tile" in cn_path.replace("\\", "/"):
                            target_path = "lllyasviel/control_v11f1e_sd15_tile"
                        
                        controlnet = ControlNetModel.from_pretrained(target_path, torch_dtype=torch.float16)
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

            try:
                # [Fix] Load as standard InpaintPipeline first to avoid config resolution errors
                self.pipe = StableDiffusionInpaintPipeline.from_single_file(ckpt_path, **load_args)
            except OSError as e:
                print(f"[ModelManager] Warning: Failed to infer config from checkpoint.")
                print(f"               Error: {e}")
                print(f"               -> This is a known Diffusers cache issue. Retrying with forced SD 1.5 config...")
                # Fallback: Force use of standard SD 1.5 Inpainting config to bypass cache/inference issues
                # Note: Using v1-5 config because the checkpoint might be a standard model (4 channels), not native inpainting (9 channels)
                self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                    ckpt_path, config="runwayml/stable-diffusion-v1-5", **load_args
                )
            
            if controlnet:
                self.pipe = StableDiffusionControlNetInpaintPipeline(**self.pipe.components, controlnet=controlnet)

            # Clip Skip 적용
            if clip_skip > 1 and hasattr(self.pipe, 'text_encoder'):
                # Diffusers 방식: 레이어 슬라이싱
                # Clip Skip 2 means removing the last 1 layer (using the 2nd to last)
                self.pipe.text_encoder.text_model.encoder.layers = self.pipe.text_encoder.text_model.encoder.layers[:-(clip_skip - 1)]

            # Move to Device
            self.pipe.to(self.device)
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.text_encoder.to(dtype=torch.float16)
            if controlnet: self.pipe.controlnet.to(dtype=torch.float16)
            if self.pipe.vae: self.pipe.vae.to(dtype=torch.float32)

            self.current_config = new_config

    def manage_lora(self, config, action="load"):
        """LoRA 주입 및 해제"""
        lora_name = config.get('lora_model', 'None')
        if lora_name == "None" or not self.pipe: return

        try:
            if action == "load":
                lora_base = cfg.get_path('lora')
                if not lora_base: return
                lora_path = os.path.join(lora_base, lora_name)
                
                if os.path.exists(lora_path):
                    self.pipe.load_lora_weights(lora_path, adapter_name="default")
                    self.pipe.fuse_lora(lora_scale=config['lora_scale'])
            
            elif action == "unload":
                self.pipe.unfuse_lora()
                self.pipe.unload_lora_weights()
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