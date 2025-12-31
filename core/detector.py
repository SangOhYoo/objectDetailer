import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# mediapipe는 선택적 의존성으로 처리 (설치 안되어 있어도 앱이 죽지 않게)
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

class ObjectDetector:
    def __init__(self, device="cuda"):
        self.device = device
        self.yolo_models = {} # Cache for YOLO models
        self.mp_face_mesh = None
        
        # 모델 저장 경로 (실제 환경에 맞게 수정 필요)
        self.model_dir = "models/detector" 
        os.makedirs(self.model_dir, exist_ok=True)

    def detect(self, image: np.ndarray, model_name: str, conf_threshold: float = 0.3) -> list:
        """
        통합 탐지 인터페이스
        Return: List of [x1, y1, x2, y2]
        """
        if "mediapipe" in model_name.lower():
            return self._detect_mediapipe(image, model_name)
        else:
            return self._detect_yolo(image, model_name, conf_threshold)

    def _detect_yolo(self, image, model_name, conf):
        # 1. Load Model (Lazy)
        if model_name not in self.yolo_models:
            model_path = os.path.join(self.model_dir, model_name)
            # 파일이 없으면 자동 다운로드 (Ultralytics 기능 활용)
            if not os.path.exists(model_path):
                print(f"[Detector] Model not found locally. Ultralytics will attempt download: {model_name}")
                model_path = model_name # 이름만 넘기면 자동 다운로드 시도함 (예: yolov8n.pt)
            
            print(f"[Detector] Loading YOLO: {model_name}")
            self.yolo_models[model_name] = YOLO(model_path)

        # 2. Predict
        model = self.yolo_models[model_name]
        results = model.predict(image, conf=conf, device=self.device, verbose=False)
        
        # 3. Parse Results
        boxes = []
        for result in results:
            for box in result.boxes:
                # xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return boxes

    def _detect_mediapipe(self, image, model_name):
        if not HAS_MEDIAPIPE:
            print("[Detector] MediaPipe not installed.")
            return []

        # 1. Init FaceMesh (Lazy)
        if self.mp_face_mesh is None:
            print("[Detector] Initializing MediaPipe FaceMesh...")
            self.mp_face_masks = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        # 2. Process
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_masks.process(rgb_image)

        boxes = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to BBox
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                
                x1 = int(min(x_coords) * w)
                y1 = int(min(y_coords) * h)
                x2 = int(max(x_coords) * w)
                y2 = int(max(y_coords) * h)
                
                # 약간의 Padding 추가 (MediaPipe는 너무 타이트하게 잡음)
                pad_x = (x2 - x1) * 0.1
                pad_y = (y2 - y1) * 0.1
                
                boxes.append([
                    int(max(0, x1 - pad_x)), 
                    int(max(0, y1 - pad_y)), 
                    int(min(w, x2 + pad_x)), 
                    int(min(h, y2 + pad_y))
                ])
        
        return boxes