# src/core/detector.py
import os
from ultralytics import YOLO

class Detector:
    def __init__(self, model_dir): self.model_dir = model_dir; self.model = None

    def detect(self, image, config):
        model_name = config.get('model', 'face_yolov8n.pt')
        if not self.model: self.model = YOLO(os.path.join(self.model_dir, model_name))
        
        results = self.model(image, conf=config.get('conf_thresh', 0.35), verbose=False)
        detections = []
        if not results: return []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask = results[0].masks.data[0].cpu().numpy() if results[0].masks else None
            # (마스크 처리 로직 생략 - 필요시 추가 가능)
            detections.append((x1, y1, x2, y2, mask if mask is not None else image[y1:y2, x1:x2])) # Placeholder mask
        return detections