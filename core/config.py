import os
import yaml

class AppConfig:
    _instance = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance.config_path = config_path
            cls._instance.load_config(config_path)
        return cls._instance

    def load_config(self, path):
        if not os.path.exists(path):
            self.data = self._get_default_config()
            print(f"[Config] Warning: Configuration file not found at {path}. Generated defaults.")
            self.save_config() # Create the file immediately
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
            # Merge with defaults to ensure all keys exist
            defaults = self._get_default_config()
            self.data = self._merge_configs(defaults, loaded)
            
        print(f"[Config] Loaded configuration from {path}")

    def _merge_configs(self, default, target):
        """Recursively merge defaults into target"""
        for k, v in default.items():
            if k not in target:
                target[k] = v
            elif isinstance(v, dict) and isinstance(target[k], dict):
                self._merge_configs(v, target[k])
        return target

    def _get_default_config(self):
        return {
            "system": {
                "gpu_detect": "cuda:0",
                "gpu_generate": "cuda:0",
                "log_level": "DEBUG",
                "max_passes": 15
            },
            "paths": {
                "checkpoint": "D:/AI_Models/Stable-diffusion",
                "sam": "D:/AI_Models/adetailer",
                "vae": "D:/AI_Models/VAE",
                "lora": "D:/AI_Models/Lora",
                "controlnet": "D:/AI_Models/ControlNet",
                "gfpgan": "D:/AI_Models/GFPGAN",
                "esrgan": "D:/AI_Models/ESRGAN"
            },
            "files": {
                "checkpoint_file": "ultra_v7.safetensors",
                "sam_file": "sam3.pt",
                "sam_config": "eval_base.yaml",
                "vae_file": "sdxl_vae.safetensors",
                "controlnet_tile": "lllyasviel/control_v11f1e_sd15_tile"
            },
            "defaults": {
                "resolution": 512,
                "padding": 1.5,
                "denoise": 0.4,
                "conf_thresh": 0.35, # Global Confidence Threshold
                "min_face_ratio": 0.01, # 1%
                "max_face_ratio": 1.0, # 100%
                "max_det": 20,
                "controlnet_weight": 0.5,
                "sort_method": "신뢰도",
                "restore_face_strength": 1.0,
                
                # [New] Hires Fix Defaults
                "hires_upscale_factor": 1.5,
                "hires_steps": 14,
                "hires_denoise": 0.4,
                "hires_cfg": 5.0,
                
                # [New] Detail Daemon Defaults
                "dd_amount": 0.1,
                "dd_start": 0.2,
                "dd_end": 0.8,
                "dd_start_offset": 0.0,
                "dd_end_offset": 0.0,
                "dd_bias": 0.5,
                "dd_exponent": 1.0,
                "dd_fade": 0.0,
                "dd_smooth": True,
                "dd_mode": "both",
                "dd_enabled": False,
                
                # [New] Soft Inpainting Defaults
                "soft_schedule_bias": 1.0,
                "soft_preservation_strength": 0.5,
                "soft_transition_contrast": 4.0,
                "soft_mask_influence": 0.0,
                "soft_diff_threshold": 0.5,
                "soft_diff_contrast": 2.0,
                
                # [New] BMAB Defaults
                "bmab_contrast": 1.0,
                "bmab_brightness": 1.0,
                "bmab_sharpness": 1.0,
                "bmab_color_saturation": 1.0,
                "bmab_color_temperature": 0.0,
                "bmab_noise_alpha": 0.0,
                "bmab_noise_alpha_final": 0.0,
                "bmab_enabled": True,
                "bmab_landscape_detail": False,
                
                # [New] BMAB Edge Defaults
                "bmab_edge_enabled": False,
                "bmab_edge_strength": 0.0,
                "bmab_edge_low": 50,
                "bmab_edge_high": 200,
                
                # [New] Canvas Expansion Defaults
                "resize_enable": False,
                "resize_ratio": 0.6,
                "resize_align": "Center",
                
                # [New] Auto-Prompting Defaults
                "auto_prompting": True,
                "interrogator_threshold": 0.35
            },
            "ui_settings": {}  # For prompt persistence
        }

    def save_config(self, new_data=None):
        """현재 설정(혹은 새로운 데이터)을 yaml 파일에 저장"""
        if new_data:
            self.data.update(new_data)
            
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            print(f"[Config] Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"[Config] Error saving configuration: {e}")
            return False

    def get_path(self, category, file_key=None):
        if not self.data or 'paths' not in self.data:
            # 경로 설정이 없을 경우를 대비한 방어 코드
            return ""
            
        base_dir = self.data['paths'].get(category, "")
        if not base_dir: return ""
            
        if file_key:
            filename = self.data.get('files', {}).get(file_key, "")
            if filename:
                return os.path.join(base_dir, filename)
        
        return base_dir

    def get(self, *keys):
        """중첩된 키 값 가져오기"""
        val = self.data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return None
            if val is None: return None
        return val

config_instance = AppConfig()