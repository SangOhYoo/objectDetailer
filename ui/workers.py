import torch
import cv2
import numpy as np
import threading
from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# --- Global Lock & Queue ---
_global_load_lock = threading.Lock() # 모델 로딩 시 충돌 방지
_job_queue = Queue() # 파일 처리 대기열

class ProcessingController(QObject):
    """
    메인 윈도우와 통신하며 GPU 워커들을 관리하는 컨트롤러
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int) # completed, total
    result_signal = pyqtSignal(str, np.ndarray) # filepath, image
    finished_signal = pyqtSignal()

    def __init__(self, file_paths, configs):
        super().__init__()
        self.file_paths = file_paths
        self.configs = configs
        self.workers = []
        self.total_files = len(file_paths)
        self.processed_count = 0

    def start_processing(self):
        # 1. 큐 채우기
        for fpath in self.file_paths:
            _job_queue.put(fpath)

        # 2. GPU 개수만큼 워커 생성 (Dual GPU = 2 workers)
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.log_signal.emit("No CUDA device found. Using CPU (Warning: Slow).")
            gpu_count = 1
        else:
            self.log_signal.emit(f"Detected {gpu_count} GPUs. Starting parallel workers...")

        # 각 GPU에 워커 할당
        for i in range(gpu_count):
            device_id = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            worker = GpuWorker(device_id, self.configs, i)
            
            # 시그널 연결
            worker.log_signal.connect(self.relay_log)
            worker.result_signal.connect(self.handle_result)
            worker.finished_signal.connect(self.check_all_finished)
            
            self.workers.append(worker)
            worker.start()

    def relay_log(self, msg):
        self.log_signal.emit(msg)

    def handle_result(self, filepath, img):
        self.processed_count += 1
        self.progress_signal.emit(self.processed_count, self.total_files)
        self.result_signal.emit(filepath, img)

    def check_all_finished(self):
        # 모든 워커가 일이 끝났는지 확인 (Queue가 비었고 워커가 쉬고있을 때)
        # 간단하게 구현: 처리된 개수가 전체 개수와 같으면 종료
        if self.processed_count >= self.total_files:
            self.log_signal.emit("All tasks completed.")
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
        self.pipe = None # Stable Diffusion Pipeline Holder
        self.detector = None # YOLO/MediaPipe Holder

    def run(self):
        self.log_signal.emit(f"[Worker-{self.worker_id}] Initialized on {self.device_id}")

        while not _job_queue.empty():
            try:
                # Non-blocking get to avoid hanging if queue empties fast
                file_path = _job_queue.get_nowait()
            except:
                break

            self.log_signal.emit(f"[Worker-{self.worker_id}] Processing: {file_path}")
            
            try:
                # 1. Load Image
                image = cv2.imread(file_path)
                if image is None:
                    self.log_signal.emit(f"[Worker-{self.worker_id}] Error loading {file_path}")
                    continue

                # 2. Process Image (The Pipeline)
                processed_image = self.process_pipeline(image)
                
                # 3. Emit Result
                self.result_signal.emit(file_path, processed_image)
                
            except Exception as e:
                self.log_signal.emit(f"[Worker-{self.worker_id}] Error: {str(e)}")
            finally:
                _job_queue.task_done()

        self.log_signal.emit(f"[Worker-{self.worker_id}] Queue empty. Worker stopping.")
        self.finished_signal.emit()

    def load_sd_model(self, model_path):
        """
        [The 'Load-Move-Cast' Pattern] - User Requirements Strict Implementation
        """
        # 이미 로드되어 있으면 패스 (Global Lock 불필요)
        if self.pipe is not None:
            return

        from diffusers import StableDiffusionInpaintPipeline

        # *** GLOBAL LOCK START ***
        with _global_load_lock:
            self.log_signal.emit(f"[Worker-{self.worker_id}] Loading model... (Holding Global Lock)")
            
            # 1. Clean up (Memory Safety)
            torch.cuda.empty_cache()

            # 2. Load Args (CPU / FP32)
            load_args = {
                "torch_dtype": torch.float32,   # 1. 무조건 FP32로 시작
                "low_cpu_mem_usage": False,     # 2. 실제 RAM에 확실히 로드
                "device_map": None,             # 3. Accelerate 간섭 차단
                "local_files_only": True        # 4. 로컬 파일 우선
            }
            
            # 로딩 (Mock path)
            # self.pipe = StableDiffusionInpaintPipeline.from_single_file(model_path, **load_args)
            # For demonstration, we simulate the object creation
            self.pipe = MagicMockPipeline() # Replace with actual load

            # 3. GPU Move
            self.pipe.to(self.device_id)

            # 4. Cast Components (FP16 Conversion except VAE)
            # if not is_sdxl: # SDXL 여부 체크 로직 필요
            if True: 
                self.pipe.unet = self.pipe.unet.to(dtype=torch.float16)
                self.pipe.text_encoder = self.pipe.text_encoder.to(dtype=torch.float16)
                self.pipe.vae = self.pipe.vae.to(dtype=torch.float32) # 7. VAE FP32 고정

            self.log_signal.emit(f"[Worker-{self.worker_id}] Model loaded on {self.device_id}. Lock released.")
        # *** GLOBAL LOCK END ***

    def process_pipeline(self, image):
        # Lazy Loading: 필요할 때 모델 로드
        self.load_sd_model("path/to/model.safetensors")

        # Mock Processing Logic similar to previous step but safely encapsulated
        # ... (Detection -> SAM -> Inpaint) ...
        
        # Inference Safety
        with torch.inference_mode():
            with torch.autocast(self.device_id.split(':')[0]): # "cuda"
                # Mock Inference
                # output = self.pipe(prompt=..., image=..., mask_image=...).images[0]
                pass
        
        return image # Return processed image

# Mock class to prevent import errors in this snippet
class MagicMockPipeline:
    def __init__(self):
        self.unet = MagicMockComponent()
        self.text_encoder = MagicMockComponent()
        self.vae = MagicMockComponent()
    def to(self, device, dtype=None):
        return self

class MagicMockComponent:
    def to(self, dtype=None):
        return self