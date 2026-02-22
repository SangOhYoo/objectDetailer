import os
import cv2
import numpy as np
import multiprocessing as mp
import shutil
import time
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# ==========================================
# Model Singletons for Worker Process
# ==========================================
_face_app = None
_nude_detector = None

def get_models(gpu_id: int):
    """
    Worker 프로세스 내에서 모델을 Singleton으로 로드합니다.
    GPU ID에 따라 CUDA Provider 설정을 제어합니다.
    """
    global _face_app, _nude_detector
    
    # GPU 할당 (환경 변수 또는 Provider 옵션)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            # InsightFace 로드
            # ctx_id: GPU ID (CUDA_VISIBLE_DEVICES로 제한했으므로 보통 0)
            _face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
        except ImportError:
            # InsightFace가 없는 경우 더미로 작동하거나 에러 발생
            _face_app = None
    
    if _nude_detector is None:
        try:
            from nudenet import NudeDetector
            # NudeNet 로드 (내부적으로 ONNX Runtime-GPU 사용)
            _nude_detector = NudeDetector()
        except ImportError:
            _nude_detector = None
        
    return _face_app, _nude_detector

# ==========================================
# Data Structures
# ==========================================
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
        self.parts: List[DetectedObject] = []
        self.gender = 'Ambiguous'
        self.centroid_x = 0
        
    def add_part(self, obj: DetectedObject):
        self.parts.append(obj)
        self._update_stats()
        
    def _update_stats(self):
        # 전체 부위의 X축 중앙값 계산
        self.centroid_x = sum(p.centroid_x for p in self.parts) / len(self.parts)
        
        # 성별 판별 로직
        # 1. 얼굴(Face)이 있으면 얼굴 성별 우선
        faces = [p for p in self.parts if p.label == 'face']
        if faces:
            # 신뢰도가 가장 높은 얼굴 기준
            best_face = max(faces, key=lambda x: x.score)
            self.gender = best_face.gender
            return

        # 2. 얼굴이 없으면 신체 부위 기준
        # Male Genitalia -> M
        # Female Genitalia / Breast -> F
        genders = [p.gender for p in self.parts if p.gender in ['M', 'F']]
        if genders:
            # 다수결 또는 우선순위 (여기서는 F 우선순위 전략 예시)
            if 'F' in genders:
                self.gender = 'F'
            else:
                self.gender = 'M'

# ==========================================
# Business Logic
# ==========================================
def cluster_objects(objects: List[DetectedObject], img_width: int) -> List[Person]:
    """
    수직 선상(X좌표 유사성)에 있는 객체들을 하나의 사람으로 묶습니다.
    (Centroid Distance algorithm)
    """
    if not objects:
        return []
    
    # X축 기준으로 정렬
    sorted_objs = sorted(objects, key=lambda x: x.centroid_x)
    
    people = []
    # X축 근접도 임계값 (이미지 너비의 12%)
    threshold = img_width * 0.12
    
    for obj in sorted_objs:
        assigned = False
        for person in people:
            if abs(person.centroid_x - obj.centroid_x) < threshold:
                person.add_part(obj)
                assigned = True
                break
        
        if not assigned:
            new_person = Person()
            new_person.add_part(obj)
            people.append(new_person)
            
    return people

def process_image_logic(img_path: str, gpu_id: int):
    """
    개별 이미지 처리 핵심 로직 (Original & 180 Rotation)
    """
    face_app, nude_det = get_models(gpu_id)
    
    img = cv2.imread(img_path)
    if img is None:
        return None, False
    
    h, w = img.shape[:2]
    
    def run_detection(image):
        detected_objs = []
        total_score = 0
        
        # 1. InsightFace 탐지
        if face_app:
            faces = face_app.get(image)
            for face in faces:
                gender = 'M' if face.gender == 1 else 'F'
                detected_objs.append(DetectedObject('face', face.bbox.tolist(), face.det_score, gender))
                total_score += face.det_score
            
        # 2. NudeNet 탐지
        if nude_det:
            nude_results = nude_det.detect(image)
            for res in nude_results:
                label = res['label']
                bbox = res['box'] # [x1, y1, x2, y2]
                score = res['score']
                
                gender = None
                if label in ['FEMALE_GENITALIA', 'FEMALE_BREAST_EXPOSED', 'FEMALE_BREAST_COVERED']:
                    gender = 'F'
                elif label in ['MALE_GENITALIA', 'MALE_BREAST']:
                    gender = 'M'
                
                detected_objs.append(DetectedObject(label, bbox, score, gender))
                total_score += score
            
        return detected_objs, total_score

    # 원본 탐지
    objs_orig, score_orig = run_detection(img)
    
    # 180도 회전 탐지
    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    objs_180, score_180 = run_detection(img_180)
    
    # 결과 비교 (신뢰도 합산 기준)
    is_upside = False
    final_objs = objs_orig
    
    # 1.1배 이상 점수가 높으면 Upside로 판단
    if score_180 > score_orig * 1.1:
        is_upside = True
        final_objs = objs_180
        for obj in final_objs:
            old_x1, old_x2 = obj.bbox[0], obj.bbox[2]
            obj.bbox[0] = w - old_x2
            obj.bbox[2] = w - old_x1
            obj.centroid_x = w - obj.centroid_x

    # 클러스터링
    people = cluster_objects(final_objs, w)
    people.sort(key=lambda x: x.centroid_x)
    
    # 패턴 생성
    pattern = "".join([p.gender[0] for p in people])
    if not pattern:
        pattern = "Unknown"
        
    return pattern, is_upside

# ==========================================
# Worker Process
# ==========================================
class ClassificationWorker(mp.Process):
    def __init__(self, gpu_id: int, job_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__()
        self.gpu_id = gpu_id
        self.job_queue = job_queue
        self.result_queue = result_queue

    def run(self):
        try:
            get_models(self.gpu_id)
            self.result_queue.put(('LOG', f"Worker {self.gpu_id} initialized on GPU {self.gpu_id}"))
        except Exception as e:
            self.result_queue.put(('LOG', f"Worker {self.gpu_id} initialization failed: {str(e)}"))
            return

        while True:
            file_path = self.job_queue.get()
            if file_path is None:
                break
            
            try:
                pattern, is_upside = process_image_logic(file_path, self.gpu_id)
                self.result_queue.put(('RESULT', {
                    'file': file_path,
                    'pattern': pattern,
                    'is_upside': is_upside
                }))
            except Exception as e:
                self.result_queue.put(('ERROR', f"Error in {file_path}: {str(e)}"))

# ==========================================
# Dispatcher (Main Controller)
# ==========================================
class Dispatcher:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.job_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []

    def start(self):
        for gpu_id in [0, 1]:
            w = ClassificationWorker(gpu_id, self.job_queue, self.result_queue)
            w.start()
            self.workers.append(w)

        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            files.extend(list(self.input_dir.rglob(ext)))
        
        if not files:
            print("[WARN] No image files found in input directory.")
            for _ in self.workers: self.job_queue.put(None)
            for w in self.workers: w.join()
            return

        for f in files:
            self.job_queue.put(str(f))

        for _ in self.workers:
            self.job_queue.put(None)

        processed_count = 0
        total_files = len(files)
        
        with tqdm(total=total_files, desc="Batch Processing") as pbar:
            while processed_count < total_files:
                msg_type, data = self.result_queue.get()
                
                if msg_type == 'LOG':
                    tqdm.write(f"[INFO] {data}")
                    continue
                elif msg_type == 'ERROR':
                    tqdm.write(f"[ERROR] {data}")
                    processed_count += 1
                    pbar.update(1)
                    continue
                
                file_path = Path(data['file'])
                pattern = data['pattern']
                is_upside = data['is_upside']
                
                target_folder = self.output_dir
                if is_upside:
                    target_folder = target_folder / "Upside"
                target_folder = target_folder / pattern
                target_folder.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.move(str(file_path), str(target_folder / file_path.name))
                except Exception as e:
                    tqdm.write(f"[ERROR] Move Failed: {file_path.name} -> {e}")
                
                processed_count += 1
                pbar.update(1)

        for w in self.workers:
            w.join()

if __name__ == "__main__":
    # Settings
    INPUT_PATH = "./input"
    OUTPUT_PATH = "./Classifier_Results"
    
    os.makedirs(INPUT_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print("\n" + "="*50)
    print(" Multi-GPU Gender Classifier Engine v1.0")
    print("="*50)
    
    # multiprocessing entry point for Windows
    mp.freeze_support()
    
    dispatcher = Dispatcher(INPUT_PATH, OUTPUT_PATH)
    dispatcher.start()
    
    print("\n[SUCCESS] Classification complete.")
