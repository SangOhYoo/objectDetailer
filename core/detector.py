import os
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

class ObjectDetector:
    def __init__(self, device="cuda", model_dir=None):
        """
        :param device: 'cuda' or 'cpu'
        :param model_dir: config.yaml에서 지정된 외부 모델 경로
        """
        self.device = device
        # model_dir이 없으면 현재 프로젝트 내 models 폴더 Fallback
        self.model_dir = model_dir if model_dir else os.path.join("models", "adetailer")
        
        self.yolo_models = {}
        self.mp_face_mesh = None

    def detect(self, image: np.ndarray, model_name: str, conf: float = 0.3) -> list:
        if "mediapipe" in model_name.lower():
            return self._detect_mediapipe(image, model_name)
        else:
            return self._detect_yolo(image, model_name, conf)

    def _detect_yolo(self, image, model_name, conf):
        if model_name not in self.yolo_models:
            # 확장자 처리
            filename = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
            model_path = os.path.join(self.model_dir, filename)

            if not os.path.exists(model_path):
                print(f"[Detector] Warning: Model not found at {model_path}. Attempting to download or use Ultralytics default.")
                load_target = filename # Ultralytics 자동 다운로드에 맡김
            else:
                load_target = model_path

            print(f"[Detector] Loading YOLO: {load_target}")
            self.yolo_models[model_name] = YOLO(load_target)

        model = self.yolo_models[model_name]
        results = model.predict(image, conf=conf, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            
            # Segmentation Masks
            masks = None
            if result.masks is not None and hasattr(result.masks, 'data'):
                masks = result.masks.data.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                det = {
                    'box': [x1, y1, x2, y2],
                    'conf': float(confs[i]),
                    'label': int(cls_ids[i]),
                    'mask': None
                }

                if masks is not None and i < len(masks):
                    raw_mask = masks[i]
                    # Resize if needed
                    if raw_mask.shape[:2] != image.shape[:2]:
                        raw_mask = cv2.resize(raw_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    binary_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                    det['mask'] = binary_mask

                detections.append(det)
        
        return detections

    def _detect_mediapipe(self, image, model_name):
        if not HAS_MEDIAPIPE:
            print("[Detector] MediaPipe not installed.")
            return []

        if self.mp_face_mesh is None:
            self.mp_face_masks = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=10, 
                refine_landmarks=True, min_detection_confidence=0.5
            )

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_masks.process(rgb)

        detections = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                
                x1, x2 = min(xs) * w, max(xs) * w
                y1, y2 = min(ys) * h, max(ys) * h
                
                # Padding (ADetailer style)
                pad_x = (x2 - x1) * 0.15
                pad_y = (y2 - y1) * 0.20
                
                box = [
                    int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)), 
                    int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
                ]
                
                detections.append({
                    'box': box, 'conf': 1.0, 'label': 0, 'mask': None
                })
        return detections