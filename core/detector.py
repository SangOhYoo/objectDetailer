import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, device_str):
        # device_str 예: "cuda:0" -> ctx_id: 0 / "cuda:1" -> ctx_id: 1
        try:
            # [수정] GPU ID 파싱 로직 강화
            if 'cuda' in device_str:
                parts = device_str.split(':')
                # 콜론 뒤에 숫자가 있으면 파싱, 없으면 0번
                ctx_id = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
            else:
                ctx_id = -1 # CPU 모드
                
            providers = ['CUDAExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
            
            # InsightFace 초기화
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            
            # ctx_id를 명시적으로 전달하여 해당 GPU 사용 강제
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            print(f"[FaceDetector] Loaded InsightFace on GPU-{ctx_id if ctx_id >=0 else 'CPU'}")
            
        except Exception as e:
            print(f"[FaceDetector] Error loading model: {e}")
            raise

    def detect(self, image):
        """
        이미지에서 얼굴을 찾아 bbox와 landmarks를 반환합니다.
        """
        # InsightFace는 내부적으로 BGR을 사용하므로 변환 불필요 (입력이 BGR이라고 가정)
        faces = self.app.get(image)
        results = []
        
        for face in faces:
            results.append({
                'bbox': face.bbox.astype(int), # [x1, y1, x2, y2]
                'kps': face.kps # 5 landmarks
            })
            
        return results