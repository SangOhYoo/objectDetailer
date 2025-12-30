import os
import sys
import torch
import numpy as np
import importlib

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sam_root_dir = os.path.join(project_root, 'sam3')
if sam_root_dir not in sys.path:
    sys.path.insert(0, sam_root_dir)

build_sam3 = None
try:
    from sam3 import model_builder
    if hasattr(model_builder, 'build_sam3'): build_sam3 = model_builder.build_sam3
    elif hasattr(model_builder, 'build_sam3_image_model'): build_sam3 = model_builder.build_sam3_image_model
    else:
        # 함수명 자동 탐색
        for name in dir(model_builder):
            if name.startswith('build_') and callable(getattr(model_builder, name)):
                build_sam3 = getattr(model_builder, name)
                break
    from sam3.model.sam3_image_predictor import Sam3ImagePredictor
except Exception as e:
    print(f"[ERROR] SAM3 Import Failed: {e}")

# core/segmentor.py

class SAMWrapper:
    def __init__(self, config_path, checkpoint_path, device_str='cuda:0'):
        # [수정] 인자로 받은 device_str을 그대로 사용하여 GPU 지정
        self.device = torch.device(device_str)
        self.predictor = None
        
        # SAM3 비활성화 안내 (현재 GPU 확인용 로그 포함)
        print(f"[SAMWrapper] SAM3 is currently DISABLED (Fallback to YOLO mask mode) on {self.device}")
        
        """ # 주석 처리된 기존 로딩 로직 (필요 시 해제)
        try:
            sam_model = build_sam3(checkpoint=checkpoint_path)
            sam_model.to(self.device)
            sam_model.eval()
            self.predictor = Sam3ImagePredictor(sam_model)
        except Exception as e:
            print(f"[SAMWrapper] Init Failed: {e}")
        """

    def set_image(self, image, caption="face"):
        if self.predictor:
            self.predictor.set_image(image, caption=caption)

    def predict_box(self, box, image_shape):
        """
        SAM3가 없으면 YOLO의 Box 영역을 꽉 채운 사각형 마스크를 생성하여 반환합니다.
        """
        if self.predictor is not None:
            masks, _, _ = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=np.array(box), multimask_output=False
            )
            return masks[0]
        else:
            # SAM3 무력화 상태: YOLO Box를 마스크로 변환
            h, w = image_shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box)
            
            # 이미지 범위 내로 좌표 클립 (안전장치)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            mask[y1:y2, x1:x2] = 255 
            return mask