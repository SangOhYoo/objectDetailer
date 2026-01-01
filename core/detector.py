import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading

try:
    from accelerate import no_dispatch
except ImportError:
    class no_dispatch:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass

try:
    import mediapipe as mp
    # [Fix] Ensure 'solutions' is available to prevent AttributeError
    if not hasattr(mp, 'solutions'):
        try:
            import mediapipe.solutions as solutions
            mp.solutions = solutions
        except ImportError:
            try:
                import mediapipe.python.solutions as solutions
                mp.solutions = solutions
            except ImportError as e_fallback:
                raise e_fallback
    HAS_MEDIAPIPE = True
except Exception as e:
    if "Descriptors cannot" in str(e):
        print("[Detector] Critical: Protobuf version conflict detected. Please run: pip install \"protobuf<5\"")
    if "No module named" in str(e):
        print("[Detector] Hint: MediaPipe might be broken. Try: pip install --force-reinstall mediapipe")
    print(f"[Detector] Warning: MediaPipe initialization failed: {e}. Landmarks disabled.")
    HAS_MEDIAPIPE = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    print("[Detector] Warning: InsightFace not installed. Gender filter disabled.")
    HAS_INSIGHTFACE = False

class ObjectDetector:
    def __init__(self, device="cuda", model_dir=None):
        self.device = device
        self.model_dir = model_dir if model_dir else os.path.join("models", "adetailer")
        self.yolo_models = {}
        self.mp_face_mesh = None
        self.face_analysis = None

    def detect(self, image: np.ndarray, model_name: str, conf: float = 0.3) -> list:
        if "mediapipe" in model_name.lower():
            return self._detect_mediapipe(image, model_name)
        else:
            return self._detect_yolo(image, model_name, conf)

    def _detect_yolo(self, image, model_name, conf):
        if model_name not in self.yolo_models:
            filename = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
            model_path = os.path.join(self.model_dir, filename)

            if not os.path.exists(model_path):
                print(f"[Detector] Warning: Model not found at {model_path}. Using Ultralytics default.")
                load_target = filename
            else:
                load_target = model_path

            print(f"[Detector] Loading YOLO: {load_target}")
            
            # [Fix] Accelerate 'init_empty_weights' hook bypass using Thread
            # Diffusers/Accelerate가 메인 스레드에 Meta Tensor Hook을 걸어두었을 가능성이 있으므로
            # 별도 스레드에서 YOLO 모델을 로드하여 Hook을 회피합니다.
            def _load_yolo():
                self.yolo_models[model_name] = YOLO(load_target, task='detect')
            
            t = threading.Thread(target=_load_yolo)
            t.start()
            t.join()

        model = self.yolo_models[model_name]
        
        # [Fix] Ultralytics device handling & Meta tensor error fallback
        device_arg = self.device
        if isinstance(device_arg, str) and device_arg.startswith("cuda:"):
            try:
                device_arg = int(device_arg.split(":")[1])
            except:
                pass
        
        try:
            with no_dispatch():
                results = model.predict(image, conf=conf, device=device_arg, verbose=False)
        except (NotImplementedError, RuntimeError) as e:
            # [Fix] Meta tensor error handling (accelerate conflict)
            print(f"[Detector] Warning: Inference failed ({e}). Attempting CPU fallback for {model_name}.")
            
            # Reload model to recover from broken state (Meta device)
            if model_name in self.yolo_models:
                del self.yolo_models[model_name]
            
            filename = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
            model_path = os.path.join(self.model_dir, filename)
            load_target = model_path if os.path.exists(model_path) else filename
            
            # [Fix] Simply reload model from path to recover from Meta tensor error
            try:
                # Threading fix for fallback as well
                fallback_result = {}
                def _reload_yolo():
                    fallback_result['model'] = YOLO(load_target, task='detect')
                t = threading.Thread(target=_reload_yolo)
                t.start()
                t.join()
                model = fallback_result.get('model')
            except Exception as e:
                print(f"[Detector] CPU Reload failed: {e}")
                return []

            self.yolo_models[model_name] = model
            try:
                with no_dispatch():
                    results = model.predict(image, conf=conf, device='cpu', verbose=False)
            except Exception as e:
                print(f"[Detector] CPU inference also failed (Skipping detection): {e}")
                return []
        
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            
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
                
                # YOLO Segmentation Mask Processing
                if masks is not None and i < len(masks):
                    raw_mask = masks[i]
                    if raw_mask.shape[:2] != image.shape[:2]:
                        raw_mask = cv2.resize(raw_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                    det['mask'] = (raw_mask > 0.5).astype(np.uint8) * 255

                detections.append(det)
        
        return detections

    def _detect_mediapipe(self, image, model_name):
        if not HAS_MEDIAPIPE:
            print("[Detector] MediaPipe is not installed.")
            return []

        if self.mp_face_mesh is None:
            # refine_landmarks=True for better eye/iris detection if needed
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=10, 
                refine_landmarks=True, min_detection_confidence=0.5
            )

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)

        detections = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. 랜드마크 좌표 추출
                points = []
                for lm in face_landmarks.landmark:
                    points.append([int(lm.x * w), int(lm.y * h)])
                points = np.array(points, dtype=np.int32)

                # 2. BBox 계산
                x1, y1 = np.min(points, axis=0)
                x2, y2 = np.max(points, axis=0)

                # Padding (ADetailer style)
                pad_x = (x2 - x1) * 0.15
                pad_y = (y2 - y1) * 0.20
                
                box = [
                    int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)), 
                    int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
                ]
                
                # 3. Convex Hull Mask 생성 (다각형 마스크)
                hull = cv2.convexHull(points)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                
                detections.append({
                    'box': box, 'conf': 1.0, 'label': 0, 
                    'mask': mask  # Polygon mask returned
                })
        return detections

    def get_face_landmarks(self, image, box):
        """
        YOLO 박스 영역 내에서 MediaPipe를 실행하여 5개 핵심 랜드마크를 추출합니다.
        반환: [[x,y], ...] (Left Eye, Right Eye, Nose, Left Mouth, Right Mouth)
        """
        if not HAS_MEDIAPIPE: return None
        
        if self.mp_face_mesh is None:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=10, 
                refine_landmarks=True, min_detection_confidence=0.5
            )
            
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # 박스보다 약간 넓게 크롭하여 랜드마크 검출 (안정성 확보)
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0: return None
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        
        if not results.multi_face_landmarks: return None
        
        # 가장 첫 번째 얼굴 사용
        face_landmarks = results.multi_face_landmarks[0]
        ch, cw = crop.shape[:2]
        
        # MediaPipe Mesh Indices: L_Eye(33), R_Eye(263), Nose(1), L_Mouth(61), R_Mouth(291)
        indices = [33, 263, 1, 61, 291]
        kps = [[int(face_landmarks.landmark[i].x * cw) + cx1, int(face_landmarks.landmark[i].y * ch) + cy1] for i in indices]
        
        return kps

    def analyze_gender(self, image, box):
        """
        InsightFace를 사용하여 박스 영역의 성별을 판별합니다.
        반환값: "Male", "Female", 또는 None (판별 불가/라이브러리 없음)
        """
        if not HAS_INSIGHTFACE:
            return None
            
        if self.face_analysis is None:
            # buffalo_l 모델 팩 사용 (가볍고 빠름)
            self.face_analysis = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        
        # 문맥 파악을 위해 박스보다 넓게 크롭
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0: return None
        
        faces = self.face_analysis.get(crop)
        if not faces: return None
        
        # 가장 큰 얼굴 기준
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        
        # InsightFace Gender: 1=Male, 0=Female
        return "Male" if face.gender == 1 else "Female"