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
        use_controlnet = controlnet_path is not None
        if not controlnet_path and cfg.get_path('controlnet', 'controlnet_tile'):
             # 기본값으로 Tile 모델 사용 (필요시)
             pass 

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
                if cn_path and os.path.exists(cn_path):
                    controlnet = ControlNetModel.from_single_file(cn_path, torch_dtype=torch.float16)

            # VAE
            vae = None
            if vae_path and os.path.exists(vae_path):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)

            # Args
            load_args = {"torch_dtype": torch.float32, "safety_checker": None, "use_safetensors": True}
            if vae: load_args["vae"] = vae
            
            if controlnet:
                PipelineClass = StableDiffusionControlNetInpaintPipeline
                load_args["controlnet"] = controlnet
            else:
                PipelineClass = StableDiffusionInpaintPipeline

            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            self.pipe = PipelineClass.from_single_file(ckpt_path, **load_args)

            # Clip Skip 적용
            if clip_skip > 1 and hasattr(self.pipe, 'text_encoder'):
                # Diffusers 방식: 레이어 슬라이싱
                self.pipe.text_encoder.text_model.encoder.layers = self.pipe.text_encoder.text_model.encoder.layers[:-clip_skip]

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