import os
import cv2
import torch
import numpy as np
import sys
import gc
from core.config import config_instance as cfg

# [Fix] torchvision 0.18+ 호환성 패치 (basicsr/gfpgan 에러 방지)
try:
    import torchvision
    import torchvision.transforms.functional as F
    # basicsr이 참조하는 구형 모듈 경로를 새 모듈로 연결
    if not hasattr(torchvision.transforms, 'functional_tensor'):
        sys.modules['torchvision.transforms.functional_tensor'] = F
    
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

        # [New] GFPGAN 및 보조 모델(facexlib) 경로 지정
        # 사용자 요청에 따라 D:/AI_Models/GFPGAN 을 최우선으로 참조합니다.
        fixed_gfpgan_root = "D:/AI_Models/GFPGAN"

        try:
            from gfpgan import GFPGANer
            
            # 1. 메인 모델 경로 결정
            # 최우선: D:/AI_Models/GFPGAN/GFPGANv1.4.pth
            model_path = os.path.join(fixed_gfpgan_root, "GFPGANv1.4.pth")
            
            # Fallback 1: 설정 파일(config.yaml)의 paths.gfpgan 을 확인
            if not os.path.exists(model_path):
                config_path = cfg.get_path('gfpgan')
                if config_path:
                    if os.path.isdir(config_path):
                        temp_path = os.path.join(config_path, "GFPGANv1.4.pth")
                    else:
                        temp_path = config_path
                    
                    if os.path.exists(temp_path):
                        model_path = temp_path
            
            # Fallback 2: 기본 models 폴더 확인
            if not os.path.exists(model_path):
                # NOTE: AppConfig.get_path does not support 'default' arg.
                model_storage = cfg.get('paths', 'model_storage_path') or "models"
                temp_path = os.path.join(model_storage, "GFPGANv1.4.pth")
                if os.path.exists(temp_path):
                    model_path = temp_path

            if not os.path.exists(model_path):
                if not self.has_warned:
                    print(f"[FaceRestorer] Warning: GFPGAN model not found at {model_path}")
                    self.has_warned = True
                return False

            # 2. 보조 모델(facexlib) 가중치 경로 설정
            # facexlib은 내부적으로 {FACEXLIB_HOME}/weights 경로를 참조합니다.
            # 사용자의 모델들이 D:/AI_Models/GFPGAN/ 에 직접 들어있으므로,
            # facexlib이 이를 인식하게 하려면 D:/AI_Models/GFPGAN/weights 폴더를 만들고 이동하거나
            # 혹은 이 코드가 실행되는 시점의 환경 변수를 조절합니다.
            
            os.environ['FACEXLIB_HOME'] = fixed_gfpgan_root
            
            self.gfpgan = GFPGANer(
                model_path=model_path, 
                upscale=1, 
                arch='clean', 
                channel_multiplier=2, 
                bg_upsampler=None, 
                device=self.device
            )
            
            print(f"[FaceRestorer] GFPGAN Loaded from: {model_path}")
            # facexlib HOME 설정 정보 출력 (사용자 안내용)
            print(f"[FaceRestorer] Facexlib Home: {fixed_gfpgan_root} (Expects models in 'weights' subfolder)")
            return True
            
        except ImportError as e:
            if not self.has_warned:
                print(f"[FaceRestorer] 'gfpgan' library import failed: {e}")
                self.has_warned = True
            return False
        except Exception as e:
            print(f"[FaceRestorer] Error loading GFPGAN: {e}")
            return False

    def unload_model(self):
        if self.gfpgan:
            print("[FaceRestorer] Unloading GFPGAN...")
            del self.gfpgan
            self.gfpgan = None
            gc.collect()
            torch.cuda.empty_cache()

    def restore(self, image_bgr, strength=0.5):
        """
        image_bgr: OpenCV BGR numpy array
        strength: 0.0 ~ 1.0 (Blend ratio, 1.0 = Full Restore)
        """
        if not self.load_model():
            return image_bgr

        _, _, output = self.gfpgan.enhance(image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        
        # [Feature] Blending for Skin Texture Control
        if strength < 1.0:
            if strength <= 0.0: return image_bgr
            
            # Linear Interpolation
            # dst = src1 * alpha + src2 * beta + gamma
            # output is 'restored', image_bgr is 'original'
            # result = restored * strength + original * (1 - strength)
            # cv2.addWeighted(src1, alpha, src2, beta, gamma)
            output = cv2.addWeighted(output, strength, image_bgr, 1.0 - strength, 0)
            
        return output