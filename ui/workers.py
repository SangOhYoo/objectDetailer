import os
import cv2
import numpy as np
import torch
import traceback
import warnings
from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# [Fix] GFPGAN(BasicSR)과 torchvision 0.17+ 호환성 패치
import torchvision
if not hasattr(torchvision.transforms, "functional_tensor"):
    import torchvision.transforms.functional as F
    import sys
    sys.modules["torchvision.transforms.functional_tensor"] = F

# [Fix] Suppress annoying FutureWarnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")

from core.pipeline import ImageProcessor  # 로직 위임
from core.config import config_instance as cfg
from core.metadata import save_image_with_metadata

class ProcessingController(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    preview_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, object)
    finished_signal = pyqtSignal()

    def __init__(self, file_paths, configs):
        super().__init__()
        self.file_paths = file_paths
        self.configs = configs
        self.processed_count = 0
        self.total_files = len(file_paths)
        self.workers = []
        self.queue = Queue()
        self.is_finished = False
        self.is_running = False

    def start_processing(self):
        for f in self.file_paths:
            self.queue.put(f)

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Detected {gpu_count} NVIDIA GPUs.")
        else:
            gpu_count = 1
            self.log_signal.emit("[System] No GPU detected. Using CPU.")

        self.is_running = True
        for i in range(gpu_count):
            dev_id = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            worker = GpuWorker(dev_id, self.configs, worker_id=i, queue=self.queue)
            worker.log_signal.connect(self.log_signal.emit)
            worker.preview_signal.connect(self.preview_signal.emit)
            worker.result_signal.connect(self.handle_result)
            worker.finished_signal.connect(self.check_finished)
            self.workers.append(worker)
            worker.start()

    def stop(self):
        """모든 워커 중지"""
        self.is_running = False
        # 큐 비우기
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except: break
        
        for worker in self.workers:
            worker.stop()
            worker.wait() # 스레드 종료 대기
        self.workers.clear()
        self.log_signal.emit("[System] Processing stopped.")

    def handle_result(self, path, img):
        self.processed_count += 1
        self.progress_signal.emit(self.processed_count, self.total_files)
        self.result_signal.emit(path, img)

    def check_finished(self):
        if self.processed_count >= self.total_files and not self.is_finished:
            self.is_finished = True
            self.finished_signal.emit()

class GpuWorker(QThread):
    log_signal = pyqtSignal(str)
    preview_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, object)
    finished_signal = pyqtSignal()

    def __init__(self, device_id, configs, worker_id, queue):
        super().__init__()
        self.device_id = device_id
        self.configs = configs
        self.worker_id = worker_id
        self.queue = queue
        self.is_stopped = False
        
        # 핵심 로직은 ImageProcessor로 이관
        self.processor = ImageProcessor(device_id, log_callback=self.relay_log, preview_callback=self.relay_preview)

    def relay_log(self, msg):
        self.log_signal.emit(f"[Worker-{self.worker_id}] {msg}")

    def relay_preview(self, img):
        # [Fix] ImageProcessor가 PIL 이미지를 보낼 경우를 대비해 Numpy(BGR) 변환
        if img is not None and not isinstance(img, np.ndarray):
            try:
                img = np.array(img)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        self.preview_signal.emit(img)

    def stop(self):
        self.is_stopped = True

    def run(self):
        self.relay_log(f"Initialized on {self.device_id}")

        while not self.queue.empty() and not self.is_stopped:
            # [Load Balancing] GPU 메모리 상태 체크하여 부하 분산
            if "cuda" in self.device_id:
                try:
                    device_idx = int(self.device_id.split(":")[1])
                    free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
                    usage_ratio = (total_mem - free_mem) / total_mem
                    
                    # 메모리 사용량이 95% 초과 시 잠시 대기 (다른 GPU가 처리하도록 유도)
                    if usage_ratio > 0.95:
                        self.relay_log(f"High VRAM usage ({usage_ratio*100:.1f}%). Yielding...")
                        self.msleep(1000) 
                        continue
                except Exception:
                    pass

            try:
                fpath = self.queue.get_nowait()
            except: break

            self.relay_log(f"Processing: {os.path.basename(fpath)}")
            try:
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    self.relay_log("Error: Load failed.")
                    self.result_signal.emit(fpath, None)
                    continue
                
                # [Fix] 실시간 진행 상황 확인을 위해 원본 이미지 먼저 전송
                self.relay_preview(img)
                
                # Processor에게 작업 위임
                result_img = self.processor.process(img, self.configs)
                
                # [New] 결과 저장
                output_dir = cfg.get('system', 'output_path') or "outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(fpath)
                save_path = os.path.join(output_dir, filename)
                
                # 메타데이터 저장을 위한 Config Wrapper
                active_conf = next((c for c in self.configs if c.get('enabled')), {})
                class ConfigWrapper:
                    def __init__(self, d): self.d = d
                    def to_adetailer_json(self): return self.d
                
                if not save_image_with_metadata(result_img, fpath, save_path, ConfigWrapper(active_conf)):
                    self.relay_log(f"Warning: Failed to save {filename}")

                self.result_signal.emit(fpath, result_img)

            except Exception as e:
                self.relay_log(f"Critical Error: {e}")
                traceback.print_exc()
                self.result_signal.emit(fpath, None)
            finally:
                self.queue.task_done()
        
        self.relay_log("Finished.")
        self.finished_signal.emit()