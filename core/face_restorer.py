import os
import cv2
import torch
import numpy as np
import sys
from core.config import config_instance as cfg

# [Fix] torchvision 0.18+ 호환성 패치 (basicsr/gfpgan 에러 방지)
try:
    import torchvision.transforms.functional as F
    if not hasattr(F, 'rgb_to_grayscale'):
        F.rgb_to_grayscale = lambda x, num_output_channels=1: F.convert_image_dtype(x, torch.float32).mean(dim=-3, keepdim=True)
except ImportError:
    pass

class FaceRestorer:
    def __init__(self, device):
        self.device = device
        self.gfpgan = None
        self.has_warned = False

    def load_model(self):
        if self.gfpgan is not None: return True
        
        # [Fix] 로컬 라이브러리 인식을 위해 프로젝트 루트를 sys.path에 추가
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from gfpgan import GFPGANer
            # 모델 경로 하드코딩 또는 설정에서 로드
            gfpgan_path = cfg.get_path('gfpgan')
            if gfpgan_path:
                if os.path.isdir(gfpgan_path):
                    model_path = os.path.join(gfpgan_path, "GFPGANv1.4.pth")
                else:
                    model_path = gfpgan_path
            else:
                model_path = os.path.join(cfg.get_path('model_storage_path', default="models"), "GFPGANv1.4.pth")

            if not os.path.exists(model_path):
                if not self.has_warned:
                    print(f"[FaceRestorer] Warning: Model not found at {model_path}")
                    self.has_warned = True
                return False
                
            self.gfpgan = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=self.device)
            print("[FaceRestorer] GFPGAN Loaded.")
            return True
        except ImportError as e:
            if not self.has_warned:
                print(f"[FaceRestorer] 'gfpgan' library import failed: {e}\n[FaceRestorer] Skipping restoration.")
                self.has_warned = True
            return False
        except Exception as e:
            print(f"[FaceRestorer] Error loading GFPGAN: {e}")
            return False

    def restore(self, image_bgr):
        """
        image_bgr: OpenCV BGR numpy array
        """
        if not self.load_model():
            return image_bgr

        _, _, output = self.gfpgan.enhance(image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        return output