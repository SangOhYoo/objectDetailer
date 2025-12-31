import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# MediaPipe Optional Import
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

class ObjectDetector:
    def __init__(self, device="cuda"):
        self.device = device
        self.yolo_models = {} 
        self.mp_face_mesh = None
        
        # 모델 저장 경로 설정
        self.model_dir = os.path.join(os.getcwd(), "models", "adetailer")
        os.makedirs(self.model_dir, exist_ok=True)

    def detect(self, image: np.ndarray, model_name: str, conf: float = 0.3) -> list:
        """
        통합 탐지 메서드
        
        Returns:
            List[dict]: [
                {
                    'box': [x1, y1, x2, y2], 
                    'mask': np.ndarray (Optional, for Seg models),
                    'conf': float,
                    'label': int
                }, ...
            ]
        """
        # MediaPipe 처리
        if "mediapipe" in model_name.lower():
            return self._detect_mediapipe(image, model_name)
        
        # YOLO 처리
        return self._detect_yolo(image, model_name, conf)

    def _download_model_if_needed(self, model_name: str) -> str:
        """
        모델 파일이 없으면 HuggingFace(Bing-su/adetailer)에서 다운로드
        """
        # 로컬 경로 확인
        local_path = os.path.join(self.model_dir, model_name)
        if os.path.exists(local_path):
            return local_path

        # 확장자가 없으면 .pt 가정
        if not model_name.endswith(".pt"):
            filename = f"{model_name}.pt"
        else:
            filename = model_name
            
        local_path = os.path.join(self.model_dir, filename)
        if os.path.exists(local_path):
            return local_path

        print(f"[Detector] Downloading model '{filename}' from HuggingFace...")
        try:
            # ADetailer 공식 Repo
            downloaded_path = hf_hub_download(
                repo_id="Bingsu/adetailer",
                filename=filename,
                local_dir=self.model_dir
            )
            return downloaded_path
        except Exception as e:
            print(f"[Detector] Download failed: {e}")
            # 다운로드 실패 시 Ultralytics 자동 다운로드에 맡기기 위해 이름만 반환
            return model_name

    def _detect_yolo(self, image, model_name, conf):
        # 1. 모델 준비
        if model_name not in self.yolo_models:
            model_path = self._download_model_if_needed(model_name)
            print(f"[Detector] Loading YOLO: {model_path}")
            self.yolo_models[model_name] = YOLO(model_path)

        model = self.yolo_models[model_name]

        # 2. 추론 (Inference)
        # classes=None이면 모든 클래스 탐지, 필요 시 [0] 등으로 필터링 가능
        # ADetailer 모델들은 대부분 class 0이 타겟임.
        results = model.predict(image, conf=conf, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            # Boxes
            boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
            confs = result.boxes.conf.cpu().numpy()  # [N]
            cls_ids = result.boxes.cls.cpu().numpy() # [N]
            
            # Segmentation Masks (Optional)
            masks = None
            if result.masks is not None:
                # [N, H, W] -> Resize masks to original image size
                # Ultralytics returns masks in original resolution if retina_masks=True(default)
                masks = result.masks.data.cpu().numpy()
                # 마스크 크기가 이미지와 다를 경우 리사이징 필요할 수 있음 (보통 Ultralytics가 처리함)
                # 여기서는 원본 이미지 크기에 맞게 가공된 마스크를 사용

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                det = {
                    'box': [x1, y1, x2, y2],
                    'conf': float(confs[i]),
                    'label': int(cls_ids[i]),
                    'mask': None
                }

                # 세그멘테이션 마스크가 있다면 처리
                if masks is not None:
                    # masks[i]는 (H, W) 크기의 float32 마스크일 수 있음
                    # YOLOv8 Seg output: 0 or 1 binary mask usually need scaling
                    raw_mask = masks[i]
                    
                    # 마스크 크기 조정 (YOLO 출력 크기가 이미지와 다를 때)
                    h, w = image.shape[:2]
                    if raw_mask.shape[:2] != (h, w):
                        raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    
                    # Binary화 (0 or 255)
                    binary_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                    det['mask'] = binary_mask

                detections.append(det)
        
        return detections

    def _detect_mediapipe(self, image, model_name):
        if not HAS_MEDIAPIPE:
            print("[Detector] MediaPipe is not installed. Skipping.")
            return []

        # FaceMesh 또는 FaceDetection 선택 (ADetailer는 주로 FaceMesh 사용)
        is_mesh = "mesh" in model_name.lower()
        
        if self.mp_face_mesh is None:
            print("[Detector] Initializing MediaPipe FaceMesh...")
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=20,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)

        detections = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. BBox 계산
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                
                x1, x2 = min(xs) * w, max(xs) * w
                y1, y2 = min(ys) * h, max(ys) * h
                
                # Padding (MediaPipe는 얼굴에 너무 딱 맞음)
                pad_x = (x2 - x1) * 0.1
                pad_y = (y2 - y1) * 0.2
                
                box = [
                    int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)),
                    int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
                ]
                
                # 2. Mask 생성 (FaceMesh의 경우 Convex Hull로 마스크 생성 가능)
                # 여기서는 간단히 None으로 두고, 필요하면 hull 계산 추가 가능
                # (ADetailer는 FaceMesh의 hull을 따서 마스크로 씀 -> 여기선 SAM이 있으니 Box만 줘도 충분)
                
                detections.append({
                    'box': box,
                    'conf': 1.0, # MediaPipe doesn't give precise conf per face in output easily
                    'label': 0,
                    'mask': None 
                })
        
        return detections