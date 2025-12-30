import numpy as np
import cv2
import os
import traceback
import torch
import queue
from threading import Lock
from PyQt6.QtCore import QThread, pyqtSignal

# [수정] 모듈화된 Core 클래스들 임포트
from core.gpu_manager import Detector as YoloDetector
from core.segmentor import SAMWrapper
from core.detector import FaceDetector as InsightFaceDetector
from core.sd_engine import SDEngine, _global_load_lock
from core.visualizer import draw_detections

class InitWorker(QThread):
    log_msg = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.log_msg.emit("[INFO] 시스템 초기화 및 GPU 상태 점검 중...")
            # [수정] 여기서 모델을 로드하면 VRAM을 선점하여 OOM을 유발하므로 제거합니다.
            # 단순히 GPU 개수만 확인합니다.
            gpu_count = torch.cuda.device_count()
            self.log_msg.emit(f"[INFO] 탐지 모델 준비 대기 중. 가용 GPU: {gpu_count}대")
            self.finished.emit()
        except Exception as e:
            self.log_msg.emit(f"[ERROR] 초기화 실패: {e}")

class ProcessWorker(QThread):
    log_msg = pyqtSignal(str)
    result_ready = pyqtSignal(object)       
    progress_update = pyqtSignal(int)       
    intermediate_update = pyqtSignal(object)

    def __init__(self, config, device_id, task_queue, active_configs):
        super().__init__()
        self.config = config
        self.device_id = device_id
        self.task_queue = task_queue
        self.active_configs = active_configs
        
        # 워커별 모델 인스턴스 보유
        self.yolo_detector = None
        self.sam_wrapper = None
        self.insight_detector = None
        self.sd_engine = None

    def run(self):
        worker_name = f"GPU-{self.device_id}"
        self.log_msg.emit(f"[{worker_name}] 워커 시작. 모델 로딩 준비...")
        
        try:
            # =========================================================
            # 1. AI 모델 초기화 (각 GPU에 독립적으로 할당)
            # =========================================================
            
            # 1-1. YOLO 초기화 (gpu_manager.py 사용)
            # YOLO 모델 경로는 첫 번째 활성 설정에서 가져오거나 기본 경로 사용
            yolo_model_dir = r"D:\AI_Models\adetailer" # 기본값 하드코딩 혹은 config에서 참조
            if self.active_configs:
                # model_path에서 디렉토리 추출
                first_model = self.active_configs[0].get('model_path', '')
                if first_model:
                    yolo_model_dir = os.path.dirname(first_model)
            
            self.yolo_detector = YoloDetector(model_dir=yolo_model_dir, device_id=self.device_id)
            
            # 1-2. SAM 초기화 (segmentor.py 사용)
            # SAM 체크포인트는 config에서 가져오거나 기본값 사용
            sam_ckpt = self.config['paths'].get('sam_checkpoint', '') # config.yaml에 정의 필요
            self.sam_wrapper = SAMWrapper(config_path=None, checkpoint_path=sam_ckpt, device_str=f"cuda:{self.device_id}")
            
            # 1-3. InsightFace 초기화 (detector.py 사용)
            # [중요] Config의 'gpu_detect'를 무시하고 현재 워커의 device_id를 강제합니다.
            self.insight_detector = InsightFaceDetector(device_str=f"cuda:{self.device_id}")

            # 1-4. SD Engine 초기화
            self.log_msg.emit(f"[{worker_name}] Stable Diffusion 엔진 로드 중...")
            self.sd_engine = SDEngine(self.config, self.device_id)
            ckpt_file = self.config['files']['checkpoint_file']
            self.sd_engine.load(ckpt_file) 
            
            self.log_msg.emit(f"[{worker_name}] 모든 모델 준비 완료. 대기열 처리 시작.")

            # =========================================================
            # 2. 큐 소비 루프
            # =========================================================
            while not self.task_queue.empty():
                try:
                    file_path = self.task_queue.get_nowait()
                except queue.Empty:
                    break 

                try:
                    file_name = os.path.basename(file_path)
                    self.log_msg.emit(f"[{worker_name}] 처리 시작: {file_name}")
                    
                    # 이미지 로드
                    img_array = np.fromfile(file_path, np.uint8)
                    current_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if current_image is None:
                        self.log_msg.emit(f"[{worker_name}] 이미지 로드 실패")
                        self.task_queue.task_done()
                        continue

                    if current_image.shape[2] == 4:
                        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGRA2BGR)

                    # === 멀티 패스 루프 ===
                    for p_idx, d_cfg in enumerate(self.active_configs):
                        pass_title = f"Pass {p_idx+1}"
                        raw_detections = []
                        
                        # A. 탐지 단계 (Detection)
                        if d_cfg['mode'] == 'yolo':
                            # gpu_manager.py의 Detector 사용
                            # config 딕셔너리에 'model' 키가 필요하므로 경로에서 파일명 추출
                            model_name = os.path.basename(d_cfg['model_path'])
                            detect_config = {
                                'model': model_name,
                                'conf_thresh': d_cfg.get('conf_thresh', 0.35)
                            }
                            
                            self.log_msg.emit(f"[{worker_name}][{pass_title}] YOLO 추론 시작 ({model_name})")
                            
                            # gpu_manager.py의 detect 메서드 호출 (device_id는 init에서 설정됨)
                            # 반환값: [(x1, y1, x2, y2, crop_img), ...]
                            yolo_results = self.yolo_detector.detect(current_image, detect_config)
                            
                            # 포맷 변환: [x1, y1, x2, y2, conf] (conf는 1.0으로 가정 혹은 YOLO 수정 필요)
                            # 현재 gpu_manager.py는 conf를 반환하지 않으므로 1.0으로 더미 처리하거나
                            # gpu_manager를 수정해야 하지만, 일단 호환성을 위해 좌표만 사용
                            for det in yolo_results:
                                x1, y1, x2, y2, _ = det
                                raw_detections.append([x1, y1, x2, y2, 0.99]) # conf 0.99 임의 지정
                            
                            self.log_msg.emit(f"   -> 감지된 객체 수: {len(raw_detections)}")

                        elif d_cfg['mode'] == 'sam':
                             # UI상 SAM3라고 되어있지만 보통 Detection은 YOLO/InsightFace가 하고 Segmentation을 SAM이 함.
                             # 여기서는 'InsightFace'를 대체제로 사용하거나, 로직에 따라 구현.
                             # 기존 코드 흐름상 'else'는 InsightFace였으므로 InsightFace 실행
                             self.log_msg.emit(f"[{worker_name}][{pass_title}] InsightFace 감지 시작")
                             raw_detections = self.insight_detector.detect(current_image)
                        
                        else:
                            self.log_msg.emit(f"[{worker_name}] 알 수 없는 모드: {d_cfg['mode']}")
                            continue

                        # 실시간 시각화 업데이트
                        if raw_detections:
                            vis_img = draw_detections(current_image, raw_detections)
                            self.intermediate_update.emit(vis_img)
                        else:
                            self.log_msg.emit(f"[{worker_name}][{pass_title}] 감지 실패 혹은 대상 없음.")
                            continue

                        # B. 인페인팅 단계 (Inpainting)
                        # geometry.py 함수들 임포트 필요 (상단에 import core.geometry 되어있어야 함)
                        from core.geometry import align_and_crop, align_and_crop_with_rotation, composite_seamless
                        
                        for i, det in enumerate(raw_detections):
                            # 포맷 정규화
                            if isinstance(det, dict): # InsightFace
                                bbox = det['bbox']
                                kps = det.get('kps', None)
                            else: # YOLO [x1, y1, x2, y2, conf]
                                bbox = det[:4]
                                kps = None

                            # 1. 크롭 및 정렬
                            if kps is not None:
                                cropped, M = align_and_crop_with_rotation(current_image, bbox, kps, target_size=512)
                            else:
                                cropped, M = align_and_crop(current_image, bbox, target_size=512)

                            # 2. SD 인페인팅 실행
                            detailed_pil = self.sd_engine.run(
                                image=cropped,
                                prompt=d_cfg.get('pos_prompt', ""),
                                neg_prompt=d_cfg.get('neg_prompt', ""), 
                                strength=d_cfg.get('denoise', 0.4), 
                                use_cnet=True,
                                hires_fix=d_cfg.get('hires_fix', False),
                                callback_on_step=None 
                            )

                            # 3. 마스크 생성 (SAM 사용 가능 시점)
                            detailed_bgr = cv2.cvtColor(np.array(detailed_pil), cv2.COLOR_RGB2BGR)
                            
                            # [옵션] SAM을 이용한 정교한 마스킹 (현재는 Box 기반 마스크 사용)
                            # 필요하다면 self.sam_wrapper.predict_box(bbox, current_image.shape) 사용 가능
                            
                            # 기본 Box 마스크
                            mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                            
                            # 4. 합성
                            M_final = M.copy()
                            if d_cfg.get('hires_fix', False):
                                M_final *= 2.0
                                M_final[2, :] = [0, 0, 1]

                            current_image = composite_seamless(current_image, detailed_bgr, mask, M_final)

                    self.result_ready.emit((file_path, current_image))
                    self.task_queue.task_done()
                    
                except Exception as e:
                    self.log_msg.emit(f"[{worker_name}] 에러 발생: {e}")
                    traceback.print_exc()
                    self.task_queue.task_done()

        except Exception as e:
            self.log_msg.emit(f"[{worker_name}] 치명적 오류 (Worker 중단): {e}")
            traceback.print_exc()
        
        finally:
            if self.sd_engine:
                del self.sd_engine
                torch.cuda.empty_cache()
            self.log_msg.emit(f"[{worker_name}] 작업 종료.")