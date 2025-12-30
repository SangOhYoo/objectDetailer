"""
core/detector.py
하이브리드 탐지 모듈 (YOLO + InsightFace)
"""

import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, config, device_id=0):
        self.config = config
        self.device_id = device_id
        self.device_str = f"cuda:{device_id}"
        
        # 1. InsightFace 로드 (성별, 랜드마크용)
        # providers 옵션을 통해 특정 GPU 할당
        providers = [('CUDAExecutionProvider', {'device_id': device_id})]
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=device_id, det_size=(640, 640))
        
        # 2. YOLO 로드 (객체/손 탐지용, 선택사항)
        self.yolo_model = None
        yolo_path = os.path.join(config.model_storage_path, config.detector_model)
        if os.path.exists(yolo_path) and "yolo" in config.detector_model.lower():
            self.yolo_model = YOLO(yolo_path)
            self.yolo_model.to(self.device_str)
            
        print(f"[Detector] Initialized on GPU {device_id}")

    def detect_faces(self, image, conf_thresh=0.5):
        """
        InsightFace를 사용하여 얼굴 탐지 + 랜드마크 + 성별 추출
        반환: List of dict
        """
        # InsightFace는 BGR 입력을 받음
        raw_faces = self.app.get(image)
        results = []
        
        for face in raw_faces:
            if face.det_score < conf_thresh:
                continue
                
            results.append({
                'bbox': face.bbox.astype(int), # [x1, y1, x2, y2]
                'kps': face.kps,               # 랜드마크 (5 points)
                'gender': face.sex,            # 'M' or 'F'
                'age': face.age,
                'score': face.det_score
            })
            
        return results

    def detect_objects(self, image, conf_thresh=0.35):
        """
        YOLO를 사용하여 일반 객체(손, 넥타이 등) 탐지
        """
        if self.yolo_model is None:
            return []
            
        results = self.yolo_model.predict(image, conf=conf_thresh, device=self.device_str, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy().astype(int),
                    'cls': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                    'label': self.yolo_model.names[int(box.cls[0])]
                })
        return detections