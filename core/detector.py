import os
import cv2
import numpy as np
import torch
import gc
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

    def offload_models(self):
        """
        [Fix] Free up VRAM by moving models to CPU or deleting them.
        """
        # 1. Access yolo_models (Thread-safe?)
        # Just clear them. YOLO loading is fast enough.
        if self.yolo_models:
            print(f"[Detector] Offloading {len(self.yolo_models)} YOLO models...")
            # Ultralytics models usually stay in VRAM.
            # Explicitly deleting might help.
            self.yolo_models.clear()
            
        # 2. InsightFace
        if self.face_analysis:
            print("[Detector] Offloading InsightFace...")
            del self.face_analysis
            self.face_analysis = None
            
        # 3. MediaPipe
        if self.mp_face_mesh:
            self.mp_face_mesh.close()
            self.mp_face_mesh = None
            
        if hasattr(self, 'mp_pose') and self.mp_pose:
            self.mp_pose.close()
            self.mp_pose = None
            
        gc.collect()
        torch.cuda.empty_cache()

    def detect(self, image: np.ndarray, model_name: str, conf: float = 0.3, classes: str = None) -> list:
        if "mediapipe" in model_name.lower():
            return self._detect_mediapipe(image, model_name)
        else:
            return self._detect_yolo(image, model_name, conf, classes)

    def _detect_yolo(self, image, model_name, conf, classes=None):
        if model_name not in self.yolo_models:
            filename = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
            model_path = os.path.join(self.model_dir, filename)
            
            # [New] External Path Support
            ext_path = os.path.join(r"D:\AI_Models\adetailer", filename)

            if os.path.exists(model_path):
                load_target = model_path
            elif os.path.exists(ext_path):
                load_target = ext_path
                print(f"[Detector] Found model in external path: {load_target}")
            else:
                print(f"[Detector] Warning: Model not found locally. Attempting auto-download/default: {filename}")
                load_target = filename

            print(f"[Detector] Loading YOLO: {load_target}")
            
            # [Fix] Direct Load (No Threading needed in Spawned Process)
            # Since we are in a clean isolated process, we don't need the thread hack
            # to bypass potential accelerate hooks from the main process.
            self.yolo_models[model_name] = YOLO(load_target)

        model = self.yolo_models[model_name]
        
        # [Fix] Ultralytics device handling & Meta tensor error fallback
        device_arg = self.device
        if isinstance(device_arg, str) and device_arg.startswith("cuda:"):
            try:
                device_arg = int(device_arg.split(":")[1])
            except:
                pass
        
        try:
            # [New] YOLO World Custom Classes Support
            # If classes string is provided (e.g., "cat, dog"), set them for YOLO World.
            # Standard YOLO models typically use 'classes' arg as efficient class filtering by INDEX.
            # But here we are dealing with Open Vocabulary names for World models.
            target_classes_arg = None
            
            if classes and isinstance(classes, str) and classes.strip():
                class_list = [c.strip() for c in classes.split(',') if c.strip()]
                
                if "world" in model_name.lower():
                     # YOLO World: We MUST set classes in the model object to define the vocabulary
                     # This persists in the model object!
                     try:
                         # Persist/Set custom vocabulary
                         model.set_classes(class_list)
                     except Exception as e:
                         print(f"[Detector] Warning: Failed to set classes for YOLO World: {e}")
                else:
                    # Standard YOLO: 'classes' arg expects list of INT indices.
                    # If user provided strings, we cannot easily map them unless we know the model's names.
                    # Try to map names to indices
                    try:
                        valid_indices = []
                        for cname in class_list:
                            for idx, mname in model.names.items():
                                if mname == cname:
                                    valid_indices.append(idx)
                        if valid_indices:
                            target_classes_arg = valid_indices
                    except:
                        pass
            
            # [New] Debugging Model Task
            print(f"[Detector] YOLO Task: {model.task}")

            # [New] Dynamic Inference Size
            # Models with '1024' or '1280' in name often require higher resolution
            infer_sz = 640
            if "1024" in model_name: infer_sz = 1024
            if "1280" in model_name: infer_sz = 1280
            
            with no_dispatch():
                results = model.predict(image, conf=conf, device=device_arg, verbose=False, classes=target_classes_arg, imgsz=infer_sz)
        except (NotImplementedError, RuntimeError) as e:
            # Conflict with accelerate / Meta tensor errors (especially for YOLOv10/v11)
            print(f"[Detector] Warning: Inference failed ({e}). Attempting CPU fallback for {model_name}.")
            try:
                # Reload model on CPU
                cpu_model = YOLO(model.ckpt_path if hasattr(model, 'ckpt_path') else model_name)
                results = cpu_model.predict(image, conf=conf, device='cpu', verbose=False, classes=target_classes_arg, imgsz=infer_sz)
            except Exception as e2:
                print(f"[Detector] Error: CPU Fallback also failed: {e2}")
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
                # Inject Class Name
                if hasattr(model, 'names') and det['label'] in model.names:
                    det['label_name'] = model.names[det['label']]
                else:
                    det['label_name'] = str(det['label'])
                
                # YOLO Segmentation Mask Processing
                if masks is not None and i < len(masks):
                    raw_mask = masks[i]
                    if raw_mask.shape[:2] != image.shape[:2]:
                        raw_mask = cv2.resize(raw_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                    det['mask'] = (raw_mask > 0.5).astype(np.uint8) * 255
                
                detections.append(det)
        
        # [Log] Detailed detection results
        if detections:
            print(f"[Detector] {model_name}: Found {len(detections)} objects.")
            for i, d in enumerate(detections):
                print(f"  #{i+1}: {d['label_name']} ({d['conf']:.2f}) at {d['box']}")
        else:
            print(f"[Detector] {model_name}: No objects found (Conf: {conf}).")

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
            
            # [Fix] Parse Device ID for InsightFace (Correctly assign GPU)
            ctx_id = 0 # Default GPU 0
            if isinstance(self.device, str) and "cuda" in self.device:
                try:
                    # cuda:1 -> 1
                    ctx_id = int(self.device.split(":")[1])
                except:
                    ctx_id = 0
            elif self.device == "cpu":
                ctx_id = -1
                
            self.face_analysis.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
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
        
        # [Modified] 가장 큰 얼굴보다는 "중심에 있는 얼굴"이 YOLO가 검출한 타겟일 확률이 높음
        # Crop의 중심점
        ccx, ccy = (crop.shape[1] // 2, crop.shape[0] // 2)
        
        def dist_from_center(f):
            # 얼굴 중심
            fcx = (f.bbox[0] + f.bbox[2]) / 2
            fcy = (f.bbox[1] + f.bbox[3]) / 2
            return (fcx - ccx)**2 + (fcy - ccy)**2

        # 중심점과의 거리가 가장 가까운 얼굴 선택
        face = min(faces, key=dist_from_center)
        
        # [Debug] 성별/나이 로그 (디버깅용)
        gender_str = "Male" if face.gender == 1 else "Female"
        # print(f"[Detector] Gender: {gender_str}, Age: {face.age}, Box: {face.bbox}") # 필요 시 주석 해제

        return gender_str

    def analyze_body_gender(self, image, box):
        """
        [New] MediaPipe Pose를 사용하여 신체 비율(어깨 vs 골반)로 성별을 추정합니다.
        얼굴이 보이지 않거나(뒷모습), InsightFace가 실패했을 때 사용됩니다.
        """
        if not HAS_MEDIAPIPE: return None

        # Lazy Loading
        if not hasattr(self, 'mp_pose') or self.mp_pose is None:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True, 
                model_complexity=1, 
                enable_segmentation=False, 
                min_detection_confidence=0.5
            )

        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # 박스보다 약간 넓게 크롭해야 어깨선이 잘 보임
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.2)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0: return None
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb)
        
        if not results.pose_landmarks: return None
        
        lm = results.pose_landmarks.landmark
        
        # 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip
        l_sh = lm[11]
        r_sh = lm[12]
        l_hip = lm[23]
        r_hip = lm[24]
        
        # 신뢰도 체크
        if min(l_sh.visibility, r_sh.visibility, l_hip.visibility, r_hip.visibility) < 0.5:
            return None # 신체가 제대로 보이지 않음

        # [Revised] 너비 계산 (Geometric - Euclidean Distance)
        # 기존 절대 좌표 차이는 회전된 신체에서 오류 발생. 유클리드 거리 사용.
        shoulder_dist = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
        hip_dist = np.linalg.norm(np.array([l_hip.x, l_hip.y]) - np.array([r_hip.x, r_hip.y]))
        
        if shoulder_dist == 0: return None
        
        ratio = hip_dist / shoulder_dist
        
        # Heuristic Threshold
        # 골반이 어깨 너비의 75% 이상이면 여성일 확률이 높음 (옷 등에 따라 다를 수 있음)
        estimated_gender = "Female" if ratio > 0.75 else "Male"
        
        # print(f"[Detector] Body Analysis - Ratio: {ratio:.2f} ({hip_dist:.2f}/{shoulder_dist:.2f}) -> {estimated_gender}")
        
        return estimated_gender

    def detect_pose(self, image: np.ndarray, conf: float = 0.3) -> list:
        """
        Detect body pose keypoints using yolov8n-pose.pt.
        Returns list of {box, keypoints} dicts.
        Keypoints format: [ [x,y,conf], ... 17 points ]
        """
        model_name = "yolo11n-pose.pt"
        if model_name not in self.yolo_models:
             filename = model_name
             model_path = os.path.join(self.model_dir, filename)
             # [New] Check External Path for Pose Model
             ext_path = os.path.join(r"D:\AI_Models\adetailer", filename)
             
             if os.path.exists(model_path):
                 load_target = model_path
             elif os.path.exists(ext_path):
                 load_target = ext_path
                 print(f"[Detector] Found Pose model in external path: {load_target}")
             else:
                 print(f"[Detector] Pose model not found, downloading {model_name}...")
                 load_target = model_name # Auto download by Ultralytics

             # Load in thread to bypass accelerate hook issues
             def _load_pose():
                self.yolo_models[model_name] = YOLO(load_target)
             t = threading.Thread(target=_load_pose)
             t.start()
             t.join()
        
        model = self.yolo_models[model_name]
        
        try:
            with no_dispatch():
                results = model.predict(image, conf=conf, verbose=False, device=self.device)
        except Exception as e:
            print(f"[Detector] Pose Detection failed: {e}")
            return []
        
        poses = []
        for r in results:
            if r.keypoints is None: continue
            
            # r.boxes contains bbox for each person
            # r.keypoints contains keypoints
            
            boxes = r.boxes.xyxy.cpu().numpy()
            kps_all = r.keypoints.data.cpu().numpy() # (N, 17, 3) -> x,y,conf
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                kps = kps_all[i]
                poses.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'keypoints': kps
                })
        return poses