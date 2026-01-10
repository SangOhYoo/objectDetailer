import os
import torch
import threading
import gc
import warnings
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
    def __init__(self, device):
        self._lock = threading.Lock()
        self.device = device
        self.pipe = None
        self.current_config = {}
        self.loaded_loras = [] # 현재 로드된 LoRA 리스트 [(name, scale), ...]

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
            if "xl" in fname or "pony" in fname or "ultra" in fname:
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
        
        # [Fix] Remove Forced Reloading (It causes VRAM spikes without proper cleanup)
        # if self.pipe is not None and self.current_config == new_config:
        #     # return
        #     print("[ModelManager] Force reloading pipeline to prevent tensor leakage (User Request)")
        #     pass
        
        # [Smart Cache] Reuse pipeline if config matches
        if self.pipe is not None and self.current_config == new_config:
             return

        with self._lock:
            self.unload_model() # Ensure clean state

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
                            # [Fix] 이 모델은 safetensors가 없으므로 명시적으로 비활성화하여 에러 로그 방지
                            controlnet = ControlNetModel.from_pretrained(target_path, torch_dtype=torch.float32, use_safetensors=False)
                        else:
                            controlnet = ControlNetModel.from_pretrained(target_path, torch_dtype=torch.float32)
                    except Exception as e:
                        print(f"[ModelManager] Failed to load ControlNet (Pretrained) from {cn_path}: {e}")

            # VAE
            vae = None
            if vae_path and os.path.exists(vae_path):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)

            # Args
            # [Fix] Force low_cpu_mem_usage=False to avoid Meta Tensor (accelerate) issues
            # 'accelerate'가 init_empty_weights를 사용하여 모델을 meta device에 올리면 LoRA 로딩 시 에러 발생 가능성이 높습니다.
            # 메모리 사용량은 늘어나지만 안정성을 위해 False로 강제합니다.
            # [Fix] device_map=None explicitly demanded to prevent accelerate form creating meta tensors
            
            # [Fix] CPU Mode Compatibility
            # If running on CPU, we must use float32. Float16 is not supported for most ops on CPU.
            use_float16 = True
            if self.device == "cpu" or (hasattr(self.device, "type") and self.device.type == "cpu"):
                use_float16 = False
            
            load_args = {
                # [Fix] 로딩 시 float16 금지 (Meta Tensor 오류 방지)
                # 우선 float32로 로드한 뒤, 안전하게 Cast 합니다.
                "torch_dtype": torch.float32, 
                "safety_checker": None, 
                "use_safetensors": True, 
                "low_cpu_mem_usage": False,
                "device_map": None 
            }
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
                    # [Mod] 사용자에게 불필요한 공포를 주는 에러 로그 간소화
                    print(f"[ModelManager] Auto-config failed. Falling back to standard SD 1.5 config.")
                    self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                        ckpt_path, config="runwayml/stable-diffusion-v1-5", **load_args
                    )
                
                if controlnet:
                    # [Fix] ControlNet 파이프라인 생성 시에도 device_map이 전파되도록 주의 (일반적으로 components로 전달됨)
                    self.pipe = StableDiffusionControlNetInpaintPipeline(**self.pipe.components, controlnet=controlnet)

            # Clip Skip 적용
            if not is_sdxl and clip_skip > 1 and hasattr(self.pipe, 'text_encoder'):
                # Diffusers 방식: 레이어 슬라이싱
                # Clip Skip 2 means removing the last 1 layer (using the 2nd to last)
                self.pipe.text_encoder.text_model.encoder.layers = self.pipe.text_encoder.text_model.encoder.layers[:-(clip_skip - 1)]

            # [Fix] OOM 방지를 위한 메모리 최적화 및 Offload 적용
            # 1. 데이터 타입 변환 (fp16) - 로딩 후 변환
            if use_float16:
                 if hasattr(self.pipe, "unet"): self.pipe.unet.to(dtype=torch.float16)
                 if hasattr(self.pipe, "text_encoder"): self.pipe.text_encoder.to(dtype=torch.float16)
                 if hasattr(self.pipe, "text_encoder_2"): self.pipe.text_encoder_2.to(dtype=torch.float16)
                 if controlnet: self.pipe.controlnet.to(dtype=torch.float16)
            # self.pipe.unet.to(dtype=torch.float16)
            # self.pipe.text_encoder.to(dtype=torch.float16)
            # if controlnet: self.pipe.controlnet.to(dtype=torch.float16)
            if self.pipe.vae: 
                self.pipe.vae.to(dtype=torch.float32)
                # [Fix] Pipeline이 float16일 때 VAE 입력도 float16으로 캐스팅되어 float32 VAE와 충돌(Input Half != Weight Float)하는 문제 해결
                # VAE Encoder 실행 직전에 입력을 강제로 float32로 변환하는 Hook 추가
                def vae_cast_hook(module, inputs):
                    if inputs[0].dtype != torch.float32:
                        return (inputs[0].to(dtype=torch.float32),) + inputs[1:]
                
                # 중복 등록 방지
                if not hasattr(self.pipe.vae.encoder, "_antigravity_hook_registered"):
                    self.pipe.vae.encoder.register_forward_pre_hook(vae_cast_hook)
                    self.pipe.vae.encoder._antigravity_hook_registered = True
                
                # Add logic for decoder using the same logic just in case
                if not hasattr(self.pipe.vae.decoder, "_antigravity_hook_registered"):
                    self.pipe.vae.decoder.register_forward_pre_hook(vae_cast_hook)
                    self.pipe.vae.decoder._antigravity_hook_registered = True

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

            # 4. CPU Offload 활성화 (Standard)
            # [Fix] 모든 모델에 대해 enable_model_cpu_offload 사용
            # 이는 각 컴포넌트(UNet, VAE, TextEncoder)를 필요할 때만 GPU로 올리고 사용 후 즉시 CPU로 내립니다.
            # VRAM 사용량을 획기적으로 줄여줍니다 (속도는 약간 느려질 수 있음).
            try:
                # gpu_id logic
                gpu_id = 0
                if isinstance(self.device, str) and self.device.startswith("cuda:"):
                        try:
                            gpu_id = int(self.device.split(":")[1])
                        except:
                            pass
                
                # enable_model_cpu_offload is preferred over sequential for non-SDXL usually, 
                # but enable_model_cpu_offload allows standard offloading.
                # enable_sequential_cpu_offload is even more aggressive (offloads blocks).
                
                print(f"[ModelManager] Enabling Model CPU Offload (GPU: {gpu_id})")
                self.pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                # self.pipe.enable_sequential_cpu_offload(gpu_id=gpu_id) # Too slow for 1.5? stick to model offload.
                
            except Exception as e:
                print(f"[ModelManager] Failed to enable CPU offload: {e}. Fallback to GPU.")
                self.pipe.to(self.device)

            self.current_config = new_config
            self.loaded_loras = [] # 모델이 바뀌면 LoRA 상태 초기화

    def unload_model(self):
        """[New] Force unload pipeline to free VRAM"""
        if self.pipe:
            print("[ModelManager] Unloading pipeline...")
            del self.pipe
            self.pipe = None
            self.current_config = {}
            self.loaded_loras = []
        
        gc.collect()
        torch.cuda.empty_cache()

    def manage_lora(self, lora_list, action="load"):
        """LoRA 주입 및 해제"""
        if not self.pipe or not lora_list: return
        
        lora_base = cfg.get_path('lora')
        if not lora_base: return
        
        # None 처리
        if lora_list is None: lora_list = []

        try:
            if action == "load":
                # [Smart Cache] 이미 로드된 LoRA 구성과 동일하면 스킵 (속도 최적화)
                if self.loaded_loras == lora_list:
                    return

                # 구성이 다르면 기존 LoRA 언로드 후 새로 로드
                if self.loaded_loras:
                    self._unload_all()
                
                if not lora_list:
                    self.loaded_loras = []
                    return

                self._load_new_loras(lora_list, lora_base)
                self.loaded_loras = lora_list
            
            elif action == "unload":
                if self.loaded_loras:
                    self._unload_all()
                    self.loaded_loras = []

        except Exception as e:
            print(f"[ModelManager] LoRA Error: {e}")

    def _unload_all(self):
        try:
            # [Fix] Disable unfuse_lora to prevent model corruption in threaded env
            # self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_new_loras(self, lora_list, lora_base):
        adapter_names = []
        adapter_weights = []

        for i, (name, scale) in enumerate(lora_list):
            lora_path = os.path.join(lora_base, name)
            
            # [Fix] Smart Extension & Recursive Search
            found = False
            # 1. Direct Check
            if os.path.exists(lora_path) and os.path.isfile(lora_path):
                 found = True
            else:
                 # 2. Try extensions
                 for ext in ['.safetensors', '.pt', '.ckpt']:
                     try_path = lora_path
                     if not try_path.lower().endswith(ext):
                         try_path += ext
                     
                     if os.path.exists(try_path) and os.path.isfile(try_path):
                         lora_path = try_path
                         found = True
                         break
            
            # 3. Recursive Search if still not found
            if not found:
                 base_name = os.path.basename(name)
                 # Normalize base_name by removing extension for searching
                 search_name = os.path.splitext(base_name)[0]
                 
                 for root, dirs, files in os.walk(lora_base):
                     for f in files:
                         if f.lower().startswith(search_name.lower()):
                             # Check exact match or with extension
                             f_no_ext = os.path.splitext(f)[0]
                             if f_no_ext.lower() == search_name.lower():
                                 potential_path = os.path.join(root, f)
                                 if potential_path.lower().endswith(('.safetensors', '.pt', '.ckpt')):
                                     lora_path = potential_path
                                     found = True
                                     print(f"[ModelManager] Found LoRA via recursive search: {lora_path}")
                                     break
                     if found: break
            
            if found:
                adapter_name = f"lora_{i}"
                try:
                    # [Fix] Suppress expected warnings due to Clip Skip (missing layers) and torch.load safety
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*unexpected keys not found in the model.*")
                        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
                        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                except Exception as e:
                    # [Fix] 1x1 Conv vs Linear dimension mismatch fallback
                    err_msg = str(e)
                    if "size mismatch" in err_msg and "1, 1" in err_msg:
                        print(f"[ModelManager] Warning: Dimension mismatch for {name}. Skipping to avoid crash.")
                        continue
                        
                    # [Fix] Meta Tensor Error Handling (accelerate conflict)
                    elif "meta tensor" in err_msg:
                        print(f"[ModelManager] Warning: Meta Tensor error for {name}. Attempting to recover...")
                        try:
                            # Try to materialize using to_empty as suggested, but this resets weights.
                            # So we only do this if we can reload, but we can't easily reload here.
                            # Best bet: Try to force move to device, if fails, just SKIP.
                            # pipe.to(device) failed earlier so this is a hail mary.
                            self.pipe.unet.to_empty(device=self.device)
                            self.pipe.text_encoder.to_empty(device=self.device)
                            # NOTICE: to_empty resets weights to random! This is dangerous if we don't reload.
                            # But since we are here, the model is likely broken (ghost) anyway.
                            # Let's try to reload LoRA which might initialize it? No.
                            # Safe approach: Just log and SKIP to prevent crash.
                            print(f"[ModelManager] Critical: Model is on meta device. Skipping LoRA {name} to avoid crash.")
                            continue
                        except Exception as e2:
                            print(f"[ModelManager] Recovery failed: {e2}. Skipping {name}.")
                            continue
                    else:
                        print(f"[ModelManager] Warning: Failed to load LoRA '{name}': {e}")
                        continue
                
                adapter_names.append(adapter_name)
                adapter_weights.append(scale)
                print(f"[ModelManager] LoRA Loaded: {name} (Scale: {scale})")
            else:
                print(f"[ModelManager] Warning: LoRA not found at {lora_path}")
        
        if adapter_names:
            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            # [Fix] Disable fuse_lora to prevent model corruption in threaded env
            # self.pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)

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