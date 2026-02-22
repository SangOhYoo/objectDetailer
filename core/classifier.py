import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Any

# ==========================================
# Model Singletons for Worker Process
# ==========================================
_face_app = None
_nude_detector = None

def get_models(gpu_id: int):
    global _face_app, _nude_detector
    
    # [Fix] Do NOT overwrite CUDA_VISIBLE_DEVICES inside the model loader
    # It should be set once at the process start.
    
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            # Only use CUDA if it's available and not forced to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            _face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            _face_app.prepare(ctx_id=0, det_size=(1280, 1280))
        except Exception as e:
            print(f"[Classifier] InsightFace Init Failed: {e}")
            _face_app = None
    
    if _nude_detector is None:
        try:
            from nudenet import NudeDetector
            _nude_detector = NudeDetector()
        except ImportError:
            _nude_detector = None
        
    return _face_app, _nude_detector

class DetectedObject:
    def __init__(self, label: str, bbox: List[float], score: float, gender: str = None):
        self.label = label
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.gender = gender # 'M', 'F', or None
        self.centroid_x = (bbox[0] + bbox[2]) / 2
        self.width = bbox[2] - bbox[0]

class Person:
    def __init__(self):
        self.parts = []
        self.gender = 'Ambiguous'
        self.centroid_x = 0
        self.total_score = 0
        
    def add_part(self, obj: DetectedObject):
        self.parts.append(obj)
        self._update_stats()
        
    def _update_stats(self):
        # Weighted centroid (based on detection score and part importance)
        # We give more weight to Face and Genitalia for centering
        total_w = sum(p.score * (10 if p.label in ['face', 'MALE_GENITALIA', 'FEMALE_GENITALIA'] else 1) for p in self.parts)
        if total_w > 0:
            self.centroid_x = sum(p.centroid_x * p.score * (10 if p.label in ['face', 'MALE_GENITALIA', 'FEMALE_GENITALIA'] else 1) for p in self.parts) / total_w
        else:
            self.centroid_x = sum(p.centroid_x for p in self.parts) / len(self.parts)
        
        # Gender Scoring (Weighted Voting)
        # [Priority] Genitalia (20) > Breast (10) > Face (5)
        m_score = 0.0
        f_score = 0.0
        
        for p in self.parts:
            weight = 1.0
            label = p.label.upper()
            if 'GENITALIA' in label: weight = 20.0
            elif 'BREAST' in label: weight = 10.0
            elif p.label == 'face': weight = 5.0
            elif 'BUTTOCKS' in label: weight = 3.0
            
            if p.gender == 'M': m_score += p.score * weight
            elif p.gender == 'F': f_score += p.score * weight
            
        if f_score > m_score and f_score > 0.05:
            self.gender = 'F'
        elif m_score > f_score and m_score > 0.05:
            self.gender = 'M'
        else:
            self.gender = 'Ambiguous'

def cluster_objects(objects: List[DetectedObject], img_width: int) -> List[Person]:
    if not objects: return []
    
    # Sort by Score DESC to use high-confidence objects as seeds
    sorted_objs = sorted(objects, key=lambda x: x.score, reverse=True)
    
    people = []
    
    for obj in sorted_objs:
        assigned = False
        # Sort current people by proximity to object centroid
        for person in sorted(people, key=lambda p: abs(p.centroid_x - obj.centroid_x)):
            # Adaptive threshold: 
            # 1. Base threshold (8% of image)
            # 2. Object's own width (people usually occupy their width)
            # 3. If it's a 'face', be more strict. If it's a body part, be more lenient.
            base_thresh = img_width * 0.08
            obj_width_thresh = obj.width * 0.7
            threshold = max(base_thresh, obj_width_thresh)
            
            # [Fix] Vertical Stack Support
            # If the object is far below/above existing parts but X is similar, it's the same person.
            dist = abs(person.centroid_x - obj.centroid_x)
            if dist < threshold:
                person.add_part(obj)
                assigned = True
                break
        
        if not assigned:
            new_person = Person()
            new_person.add_part(obj)
            people.append(new_person)
            
    return people

def process_image_classification(img_path: str, gpu_id: int):
    face_app, nude_det = get_models(gpu_id)
    
    # [Fix] Robust way to load image with non-ASCII paths on Windows
    try:
        img_stream = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Classifier] Failed to load {img_path}: {e}")
        return "Error", False
        
    if img is None: 
        print(f"[Classifier] Decoded image is None: {img_path}")
        return "Error", False
    
    h, w = img.shape[:2]
    
    def run_detection(image):
        detected_objs = []
        face_count = 0
        total_score = 0
        
        # 1. InsightFace
        if face_app:
            try:
                faces = face_app.get(image)
                for face in faces:
                    if face.det_score < 0.4: continue
                    gender = 'M' if face.gender == 1 else 'F'
                    detected_objs.append(DetectedObject('face', face.bbox.tolist(), face.det_score, gender))
                    total_score += face.det_score * 3.0
                    face_count += 1
            except Exception as e:
                print(f"[Classifier] InsightFace error: {e}")
            
        # 2. NudeNet
        if nude_det:
            try:
                nude_results = nude_det.detect(image)
                for res in nude_results:
                    if res['score'] < 0.3: continue
                    label = res['label']
                    bbox = res['box']
                    score = res['score']
                    
                    gender = None
                    if label in ['FEMALE_GENITALIA', 'FEMALE_BREAST_EXPOSED', 'FEMALE_BREAST_COVERED', 'BUTTOCKS_EXPOSED', 'BUTTOCKS_COVERED']:
                        gender = 'F'
                    elif label in ['MALE_GENITALIA', 'MALE_BREAST']:
                        gender = 'M'
                    
                    detected_objs.append(DetectedObject(label, bbox, score, gender))
                    total_score += score
            except Exception as e:
                print(f"[Classifier] NudeNet error: {e}")
                
        return detected_objs, total_score, face_count

    # 원본 탐지
    objs_orig, score_orig, faces_orig = run_detection(img)
    
    # 180도 회전 탐지
    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    objs_180, score_180, faces_180 = run_detection(img_180)
    
    # Orientation Logic - More robust
    is_upside = False
    final_objs = objs_orig
    
    # Case 1: 얼굴이 회전했을 때만 탐지되는 경우 (가장 확실함)
    if faces_180 > 0 and faces_orig == 0:
        is_upside = True
    # Case 2: 회전했을 때 탐지 점수 합계가 월등히 높은 경우
    elif score_180 > score_orig * 1.5:
        is_upside = True
    # Case 3: 양쪽 다 얼굴이 있지만 회전했을 때 얼굴 점수 합계가 높은 경우
    elif faces_180 > 0 and faces_orig > 0:
        face_score_orig = sum(p.score for p in objs_orig if p.label == 'face')
        face_score_180 = sum(p.score for p in objs_180 if p.label == 'face')
        if face_score_180 > face_score_orig + 0.2:
            is_upside = True

    if is_upside:
        final_objs = objs_180
        for obj in final_objs:
            old_x1, old_x2 = obj.bbox[0], obj.bbox[2]
            obj.bbox[0] = w - old_x2
            obj.bbox[2] = w - old_x1
            obj.centroid_x = w - obj.centroid_x

    # 클러스터링
    people = cluster_objects(final_objs, w)
    
    # X축 정렬
    people.sort(key=lambda x: x.centroid_x)
    
    # 패턴 생성
    valid_persons = []
    for p in people:
        # Minimum score threshold for a valid person (e.g. sum of scores > 0.4)
        total_p_score = sum(part.score for part in p.parts)
        if total_p_score > 0.45:
            valid_persons.append(p)
    
    pattern = "".join([p.gender[0] for p in valid_persons])
    if not pattern: pattern = "Unknown"
    return pattern, is_upside
