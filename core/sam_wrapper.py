import numpy as np
import torch
import cv2
import os

# segment_anything 라이브러리가 설치되어 있어야 합니다.
# pip install segment-anything
from segment_anything import sam_model_registry, SamPredictor

class SamInference:
    def __init__(self, model_type="vit_b", checkpoint="models/sam_vit_b_01ec64.pth", device="cuda"):
        """
        SAM 모델 초기화
        :param model_type: vit_b (Basic), vit_l (Large), vit_h (Huge)
        :param checkpoint: 모델 가중치 파일 경로
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.predictor = None
        self.is_loaded = False
        self.is_image_set = False

    def load_model(self):
        if self.is_loaded:
            return

        if not os.path.exists(self.checkpoint):
            print(f"[SAM] Warning: Checkpoint not found at {self.checkpoint}")
            return

        print(f"[SAM] Loading {self.model_type} model to {self.device}...")
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        self.is_loaded = True

    def set_image(self, image_bgr):
        """
        이미지 임베딩 생성 (한 번만 실행하면 여러 박스에 대해 추론 가능)
        """
        if not self.is_loaded:
            self.load_model()
        
        if self.predictor is None:
            raise RuntimeError("SAM model failed to load.")

        # SAM은 RGB 입력을 받음
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        self.is_image_set = True

    def predict_mask_from_box(self, box):
        """
        Box 프롬프트를 사용하여 마스크 생성
        box: [x1, y1, x2, y2] format
        """
        if not self.is_image_set:
            raise RuntimeError("Set image first using set_image()")

        # SAM expects numpy array for box
        input_box = np.array(box)

        # Predict
        # point_coords=None, point_labels=None since we use box prompt
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :], # Add batch dim
            multimask_output=False  # 가장 신뢰도 높은 마스크 1개만 리턴
        )
        
        # masks shape: (1, H, W) -> (H, W) uint8 0-255
        mask = (masks[0] * 255).astype(np.uint8)
        return mask