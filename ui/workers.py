import os
import cv2
import numpy as np
import torch
import traceback
from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from core.pipeline import ImageProcessor  # 로직 위임

_job_queue = Queue()

class ProcessingController(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    result_signal = pyqtSignal(str, np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, file_paths, configs):
        super().__init__()
        self.file_paths = file_paths
        self.configs = configs
        self.processed_count = 0
        self.total_files = len(file_paths)
        self.workers = []

    def start_processing(self):
        for f in self.file_paths:
            _job_queue.put(f)

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Detected {gpu_count} NVIDIA GPUs.")
        else:
            gpu_count = 1
            self.log_signal.emit("[System] No GPU detected. Using CPU.")

        for i in range(gpu_count):
            dev_id = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            worker = GpuWorker(dev_id, self.configs, worker_id=i)
            worker.log_signal.connect(self.log_signal.emit)
            worker.result_signal.connect(self.handle_result)
            worker.finished_signal.connect(self.check_finished)
            self.workers.append(worker)
            worker.start()

    def handle_result(self, path, img):
        self.processed_count += 1
        self.progress_signal.emit(self.processed_count, self.total_files)
        self.result_signal.emit(path, img)

    def check_finished(self):
        if self.processed_count >= self.total_files:
            self.finished_signal.emit()

class GpuWorker(QThread):
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str, np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, device_id, configs, worker_id):
        super().__init__()
        self.device_id = device_id
        self.configs = configs
        self.worker_id = worker_id
        
        # 핵심 로직은 ImageProcessor로 이관
        self.processor = ImageProcessor(device_id, log_callback=self.relay_log)

    def relay_log(self, msg):
        self.log_signal.emit(f"[Worker-{self.worker_id}] {msg}")

    def run(self):
        self.relay_log(f"Initialized on {self.device_id}")

        while not _job_queue.empty():
            try:
                fpath = _job_queue.get_nowait()
            except: break

            self.relay_log(f"Processing: {os.path.basename(fpath)}")
            try:
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    self.relay_log("Error: Load failed.")
                    continue
                
                # Processor에게 작업 위임
                result_img = self.processor.process(img, self.configs)
                self.result_signal.emit(fpath, result_img)

            except Exception as e:
                self.relay_log(f"Critical Error: {e}")
                traceback.print_exc()
            finally:
                _job_queue.task_done()
        
        self.relay_log("Finished.")
        self.finished_signal.emit()