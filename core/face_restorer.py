import os
import cv2
import torch
import numpy as np

class FaceRestorer:
    def __init__(self, device):
        self.device = device
        self.gfpgan = None
        self.has_warned = False

    def load_model(self):
        if self.gfpgan is not None: return True
        
        try:
            from gfpgan import GFPGANer
            # 모델 경로 하드코딩 또는 설정에서 로드
            model_path = os.path.join("models", "GFPGANv1.4.pth")
            if not os.path.exists(model_path):
                if not self.has_warned:
                    print(f"[FaceRestorer] Warning: Model not found at {model_path}")
                    self.has_warned = True
                return False
                
            self.gfpgan = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=self.device)
            print("[FaceRestorer] GFPGAN Loaded.")
            return True
        except ImportError:
            if not self.has_warned:
                print("[FaceRestorer] 'gfpgan' library not installed. Skipping restoration.")
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