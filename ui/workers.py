"""
ui/workers.py
SAM3_FaceDetailer_Ultimate 작업 관리자
- 듀얼 GPU 독립 큐(Queue) 소비
- 회전 보정(Alignment) 및 복원(Restore) 파이프라인 제어
- PyQt6 시그널 통신
"""

import cv2
import numpy as np
import torch
import traceback
import queue
import os
from PyQt6.QtCore import QThread, pyqtSignal

# [New] 우리가 작성한 핵심 모듈 임포트
from configs import DetailerConfig, SystemConfig
from core.detector import FaceDetector
from core.sd_engine import SDEngine
from core.geometry import align_and_crop, restore_and_paste
from core.metadata import save_image_with_metadata

class InitWorker(QThread):
    """
    프로그램 시작 시 GPU 상태를 점검하는 가벼운 워커
    """
    log_msg = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, sys_config: SystemConfig):
        super().__init__()
        self.sys_config = sys_config

    def run(self):
        try:
            self.log_msg.emit("[System] GPU 상태 점검 중...")
            if torch.cuda.is_available():
                cnt = torch.cuda.device_count()
                self.log_msg.emit(f"[System] 발견된 GPU: {cnt}대")
                for i in range(cnt):
                    props = torch.cuda.get_device_properties(i)
                    self.log_msg.emit(f"   - GPU {i}: {props.name} (VRAM: {props.total_memory / 1024**3:.1f} GB)")
            else:
                self.log_msg.emit("[Warning] CUDA를 찾을 수 없습니다. CPU 모드로 동작합니다.")
            
            self.finished.emit()
        except Exception as e:
            self.log_msg.emit(f"[Error] 초기화 중 오류: {e}")
            self.finished.emit()


class ProcessWorker(QThread):
    """
    [핵심] 실제 이미지를 처리하는 워커 스레드 (GPU 당 1개씩 생성됨)
    """
    log_msg = pyqtSignal(str)             # 로그 메시지 전송
    progress_update = pyqtSignal(int)     # 진행률 업데이트
    result_ready = pyqtSignal(str)        # 처리 완료 신호 (파일명)
    error_occurred = pyqtSignal(str)      # 에러 발생 신호

    def __init__(self, device_id, task_queue, sys_config: SystemConfig):
        super().__init__()
        self.device_id = device_id
        self.task_queue = task_queue
        self.sys_config = sys_config
        self.is_running = True
        
        # 엔진 인스턴스 (run 메서드에서 초기화)
        self.detector = None
        self.sd_engine = None

    def run(self):
        worker_name = f"GPU-{self.device_id}"
        self.log_msg.emit(f"[{worker_name}] 워커 시작. 엔진 초기화 중...")

        try:
            # 1. AI 엔진 로드 (설정은 configs.py 기본값 참조하거나 여기서 주입)
            # (1) 탐지기 로드 (InsightFace + YOLO)
            self.detector = FaceDetector(self.sys_config, device_id=self.device_id)
            
            # (2) SD 엔진 로드 (빈 껍데기 생성 후 첫 작업 때 모델 로드)
            self.sd_engine = SDEngine(self.sys_config, device_id=self.device_id)
            
            self.log_msg.emit(f"[{worker_name}] 엔진 준비 완료. 대기열 감시 시작.")

            # 2. 작업 큐 소비 루프 (Infinite Loop)
            while self.is_running:
                try:
                    # 큐에서 작업 가져오기 (1초 대기 타임아웃)
                    # task는 (image_path, detailer_config) 튜플 형태여야 함
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 작업 시작
                image_path, config = task
                file_name = os.path.basename(image_path)
                
                try:
                    self.process_image(image_path, config, worker_name)
                    self.task_queue.task_done()
                    self.result_ready.emit(file_name)
                    
                except Exception as e:
                    self.log_msg.emit(f"[{worker_name}] 처리 실패 ({file_name}): {e}")
                    traceback.print_exc()
                    self.task_queue.task_done()
                    
                    # [좀비 모드] 치명적 에러(OOM 등) 시 복구 시도
                    if "out of memory" in str(e).lower() and self.sys_config.auto_recover:
                        self.log_msg.emit(f"[{worker_name}] ⚠️ OOM 감지! 메모리 정리 및 재부팅...")
                        self.sd_engine._cleanup()
                        torch.cuda.empty_cache()
                        
        except Exception as e:
            self.log_msg.emit(f"[{worker_name}] 워커 치명적 오류로 종료: {e}")
        finally:
            self.log_msg.emit(f"[{worker_name}] 워커 종료됨.")

    def process_image(self, image_path, config: DetailerConfig, worker_name):
        """
        이미지 1장을 처리하는 전체 파이프라인
        Detect -> Crop & Align -> Inpaint -> Restore -> Save
        """
        # 1. 이미지 로드 (한글 경로 대응)
        try:
            img_array = np.fromfile(image_path, np.uint8)
            full_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if full_image is None:
                raise ValueError("이미지 데이터가 비어있습니다.")
        except Exception as e:
            raise ValueError(f"이미지 로드 실패: {e}")

        # 2. 모델 체크포인트 로드 (필요 시 교체)
        # SDEngine 내부 캐싱 로직 활용
        ckpt_path = os.path.join(self.sys_config.model_storage_path, config.checkpoint_file)
        # 현재 로드된 모델과 다르면 로드 (SDEngine 내부에서 판단 권장하지만 여기서 명시적 호출)
        if self.sd_engine.pipe is None: 
             self.log_msg.emit(f"[{worker_name}] 모델 로딩: {config.checkpoint_file}")
             self.sd_engine.load_model(ckpt_path)

        # 3. 얼굴 탐지 (Detection)
        faces = self.detector.detect_faces(full_image, conf_thresh=config.conf_thresh)
        self.log_msg.emit(f"[{worker_name}] {len(faces)}명 감지됨 ({os.path.basename(image_path)})")

        final_image = full_image.copy()
        
        # 4. 각 얼굴별 처리 루프
        for i, face in enumerate(faces):
            bbox = face['bbox']
            kps = face['kps']
            
            # [Smart Filter] 너무 작은 얼굴 건너뛰기
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_area = final_image.shape[0] * final_image.shape[1]
            if (face_area / total_area) < config.min_face_ratio:
                continue

            # 5. 잘라내기 및 회전 보정 (Crop & Align)
            # auto_rotate=True면 눈 좌표 기반으로 0도로 세움
            aligned_crop, M = align_and_crop(
                final_image, 
                bbox, 
                kps=kps if config.auto_rotate else None, 
                target_size=config.target_res,
                padding=config.crop_padding,
                force_rotate=config.auto_rotate
            )

            # 6. 프롬프트 구성 (성별 등 자동 주입)
            current_prompt = config.pos_prompt
            if config.auto_prompt_injection:
                # InsightFace 성별: 1=Male, 0=Female
                gender_tag = "1boy, male" if face['gender'] == 1 else "1girl, female"
                current_prompt = f"{gender_tag}, {current_prompt}"

            # 7. 인페인팅 (Inpainting)
            # SD Engine에게 '정자세 얼굴'을 넘겨줌
            processed_crop = self.sd_engine.run(
                image=aligned_crop,
                prompt=current_prompt,
                neg_prompt=config.neg_prompt,
                strength=config.denoising_strength,
                seed=config.seed,
                guidance_start=config.guidance_start,
                guidance_end=config.guidance_end
            )

            # 8. 역회전 및 합성 (Restore & Paste)
            final_image = restore_and_paste(
                final_image, 
                processed_crop, 
                M, 
                mask_blur=config.mask_blur
            )

        # 9. 결과 저장 (메타데이터 보존)
        save_name = f"result_{os.path.basename(image_path)}"
        save_full_path = os.path.join(self.sys_config.output_path, save_name)
        os.makedirs(self.sys_config.output_path, exist_ok=True)

        # 메타데이터 포함 저장 (PIL 사용)
        success = save_image_with_metadata(final_image, image_path, save_full_path, config)
        
        if not success:
            self.log_msg.emit(f"[{worker_name}] 메타데이터 저장 실패 -> 일반 저장으로 대체됨")

    def stop(self):
        self.is_running = False
        self.wait()