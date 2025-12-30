# core/gpu_manager.py

import os
import cv2
import torch
from ultralytics import YOLO

class Detector:
    def __init__(self, model_dir, device_id=0): 
        """
        [수정] 초기화 시 할당받은 GPU ID를 저장합니다.
        """
        self.model_dir = model_dir
        self.device_str = f"cuda:{device_id}" # 예: "cuda:0" 또는 "cuda:1"
        self.model = None

    def detect(self, image, config):
        model_name = config.get('model', 'face_yolov8n.pt')
        
        if not self.model: 
            model_path = os.path.join(self.model_dir, model_name)
            # [수정] 모델 로드 직후 해당 GPU로 이동
            self.model = YOLO(model_path)
            self.model.to(self.device_str)
        
        # 4채널(RGBA) 이미지를 3채널(BGR)로 변환
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # [수정] 추론 실행 시 device 명시하여 GPU 분산 처리 강제
        results = self.model(
            image, 
            conf=config.get('conf_thresh', 0.35), 
            verbose=False,
            device=self.device_str 
        )
        
        detections = []
        if not results or len(results) == 0: 
            return []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 원본 이미지에서 해당 영역 크롭 및 저장
            detections.append((x1, y1, x2, y2, image[y1:y2, x1:x2])) 
            
        return detections