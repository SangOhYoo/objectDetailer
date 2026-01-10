import os
import cv2
import numpy as np
import torch
import traceback
import warnings
import multiprocessing as mp
import time
from queue import Empty
import gc
from PyQt6.QtCore import pyqtSignal, QObject, QTimer

# [Fix] GFPGAN(BasicSR)과 torchvision 0.17+ 호환성 패치
import torchvision
if not hasattr(torchvision.transforms, "functional_tensor"):
    import torchvision.transforms.functional as F
    import sys
    sys.modules["torchvision.transforms.functional_tensor"] = F

# [Fix] Suppress annoying FutureWarnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")

from core.pipeline import ImageProcessor
from core.config import config_instance as cfg
from core.metadata import save_image_with_metadata

def worker_process(device_id, input_queue, output_queue, configs):
    """
    별도의 프로세스에서 실행되는 워커 함수.
    QThread가 아닌 multiprocessing.Process로 실행됨.
    """
    worker_id = device_id
    
    # [Fix] Force-clear CUDA_VISIBLE_DEVICES to ensure multiple GPUs are visible
    # On Windows/Multiprocessing, sometimes this env var gets inherited or set implicitly to 0.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    # [Log] Immediate process startup check
    print(f"[Process Start] Worker PID: {os.getpid()}, Req Device: {device_id}")

    try:
        # [MP] 프로세스 별로 ImageProcessor 초기화 (독립 메모리)
        # 로그는 큐를 통해 메인 프로세스로 전달
        def log_wrapper(msg):
            output_queue.put(("log", f"[{worker_id}] {msg}"))
            
        def preview_wrapper(img):
            # ... existing functionality ...
            if img is not None and not isinstance(img, np.ndarray):
                try:
                    img = np.array(img)
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except:
                    pass
            output_queue.put(("preview", img))

        log_wrapper(f"Worker Process Started (PID: {os.getpid()})")
        
        if "cuda" in device_id:
            try:
                # [Debug] Worker Process GPU Diagnostics
                # Force torch to initialize CUDA to check actual visibility
                if not torch.cuda.is_available():
                     print(f"[Worker] torch.cuda.is_available() returned False inside worker {worker_id}")
                     raise RuntimeError("CUDA not available in worker process")

                local_device_count = torch.cuda.device_count()
                req_idx = int(device_id.split(':')[1])
                
                print(f"[Worker {worker_id}] Diagnostic Start:")
                print(f"  - PID: {os.getpid()}")
                print(f"  - Device Count: {local_device_count}")
                
                # [Fix] Adaptive Device Selection
                # If we see multiple GPUs, trust the requested index (e.g. cuda:1 -> cuda:1).
                # If we see only 1 GPU (Isolation), map any request to cuda:0.
                final_device_id = device_id
                
                if req_idx < local_device_count:
                    final_device_id = f"cuda:{req_idx}"
                elif local_device_count == 1:
                    print(f"[Worker] Warning: Requested {device_id} but only 1 device visible. Mapping to cuda:0.")
                    final_device_id = "cuda:0"
                else:
                    raise RuntimeError(f"Requested {device_id} but only {local_device_count} devices found.")
                
                device_id = final_device_id

                # [Fix] Explicitly set device for this process context
                torch.cuda.set_device(device_id)
                current_dev = torch.cuda.current_device()
                current_dev_name = torch.cuda.get_device_name(current_dev)
                
                msg = f"GPU Bound: {device_id} -> Actual: {current_dev} ({current_dev_name})"
                print(f"[Worker] {msg}")
                log_wrapper(msg) # Send to UI log as well

                t = torch.tensor([1.0]).to(device_id)
                del t
                print(f"[Worker] GPU Check Passed for {device_id} (Originally: {worker_id})")
            except Exception as e:
                err_msg = f"Error: Device {device_id} is invalid ({e}). Falling back to CPU."
                log_wrapper(err_msg)
                print(f"[Worker Error] {err_msg}") # Print to terminal for visibility
                device_id = "cpu"

        processor = ImageProcessor(device_id, log_callback=log_wrapper, preview_callback=preview_wrapper)
        log_wrapper(f"Initialized ImageProcessor on {device_id}")

        while True:
            try:
                # 큐에서 작업 가져오기 (타임아웃 1초)
                task = input_queue.get(timeout=1.0)
            except Empty:
                # 타임아웃 발생 시 다시 루프 (종료 시그널 확인 등 필요 시 확장 가능)
                continue
                
            if task is None: # 종료 시그널
                break
                
            # [Fix] Handle (path, angle) or path
            if isinstance(task, tuple):
                fpath, angle = task
            else:
                fpath, angle = task, 0
                
            output_queue.put(("started", os.path.basename(fpath)))
            log_wrapper(f"Processing: {os.path.basename(fpath)} (Rot: {angle})")
            
            try:
                # 이미지 로드
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    log_wrapper("Error: Load failed.")
                    output_queue.put(("result", (fpath, None)))
                    continue
                
                # [New] Apply Input Rotation (Pre-Process)
                if angle != 0:
                    if angle == 90 or angle == -270:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180 or angle == -180:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif angle == 270 or angle == -90:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # 프리뷰 전송
                preview_wrapper(img)
                
                # 처리 실행
                result_img = processor.process(img, configs)
                
                # [New] Apply Output Inverse Rotation (Restore Original Angle)
                # Only if result exists and matches dimensions (or just force rotate)
                if angle != 0 and result_img is not None:
                     # Inverse
                     # 90 (CW) -> -90 (CCW)
                     if angle == 90 or angle == -270:
                         result_img = cv2.rotate(result_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                     elif angle == 180 or angle == -180:
                         result_img = cv2.rotate(result_img, cv2.ROTATE_180)
                     elif angle == 270 or angle == -90:
                         result_img = cv2.rotate(result_img, cv2.ROTATE_90_CLOCKWISE)

                # 결과 저장 로직 (프로세스 내부에서 저장까지 완료)
                # Config가 Pickling되어 넘어왔으므로 여기서 cfg 인스턴스 접근 시 주의 필요
                # 하지만 save_image_with_metadata는 경로만 알면 됨. output_path는 다시 읽거나 전달받아야 함.
                # 편의상 cfg를 여기서 다시 로드하거나 safe한 경로 사용
                # 여기서는 메인 프로세스에서 전달된 configs를 사용
                
                output_dir = "outputs" # Default
                if cfg.get('system', 'output_path'):
                    output_dir = cfg.get('system', 'output_path')
                
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(fpath)
                save_path = os.path.join(output_dir, filename)
                
                # [Modified] Sync Save (No threading)
                # Removed ThreadPoolExecutor to ensure stability and process-only architecture.
                
                # 메타데이터 저장용 래퍼
                active_conf = next((c for c in configs if c.get('enabled')), {})
                class ConfigWrapper:
                    def __init__(self, d): self.d = d
                    def to_adetailer_json(self): return self.d
                
                try:
                    save_success = save_image_with_metadata(result_img, fpath, save_path, ConfigWrapper(active_conf))
                    if not save_success:
                        log_wrapper(f"Warning: Failed to save metadata for {filename}")
                except Exception as e:
                    log_wrapper(f"Error saving {filename}: {e}")

                # 결과 이미지(Numpy) 전송
                output_queue.put(("result", (fpath, result_img)))

            except Exception as e:
                log_wrapper(f"Critical Error: {e}")
                traceback.print_exc()
                output_queue.put(("result", (fpath, None)))
            finally:
                # [Optimized] 불필요한 매 루프마다의 GC/EmptyCache 제거 (속도 향상)
                # PyTorch 캐시 얼로케이터를 활용하여 재할당 오버헤드 감소
                pass

    except Exception as e:
        print(f"Worker Process Died: {e}")
        traceback.print_exc()

class ProcessingController(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    file_started_signal = pyqtSignal(str)
    preview_signal = pyqtSignal(np.ndarray)
    result_signal = pyqtSignal(str, object)
    finished_signal = pyqtSignal()

    def __init__(self, file_paths, configs):
        super().__init__()
        self.file_paths = file_paths
        self.configs = configs
        self.processes = []
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        
        # [New] QTimer for polling instead of QThread
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_queue)
        self.total_files = len(file_paths)
        self.processed_count = 0

    def start_processing(self):
        # 작업 큐 채우기
        for f in self.file_paths:
            self.input_queue.put(f)

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Detected {gpu_count} NVIDIA GPUs. Starting Multiprocessing...")
            for i in range(gpu_count):
                try:
                    name = torch.cuda.get_device_name(i)
                    self.log_signal.emit(f"  - GPU {i}: {name}")
                except:
                    self.log_signal.emit(f"  - GPU {i}: Unknown Device")
        else:
            gpu_count = 1
            self.log_signal.emit("[System] No GPU detected. Using CPU via Multiprocessing.")

        # 워커 프로세스 시작
        for i in range(gpu_count):
            dev_id = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            p = mp.Process(
                target=worker_process, 
                args=(dev_id, self.input_queue, self.output_queue, self.configs),
                daemon=True # 메인 프로세스 종료 시 함께 종료
            )
            p.start()
            self.processes.append(p)
            self.log_signal.emit(f"[Manager] Started Worker Process {i} (PID: {p.pid}) on {dev_id}")
            
        # Start Polling Timer (Time interval 50ms)
        self.timer.start(50)

    def check_queue(self):
        """
        Polls the output queue for messages from worker processes.
        Runs on the Main Thread via QTimer.
        """
        # Process multiple messages per tick to prevent backlog
        # But limit to avoid freezing UI if flooded
        for _ in range(20): 
            try:
                # Non-blocking get
                msg_type, data = self.output_queue.get_nowait()
                
                if msg_type == "log":
                    self.log_signal.emit(data)
                elif msg_type == "preview":
                    self.preview_signal.emit(data)
                elif msg_type == "started":
                    self.file_started_signal.emit(data)
                elif msg_type == "result":
                    path, img = data
                    self.processed_count += 1
                    self.progress_signal.emit(self.processed_count, self.total_files)
                    self.result_signal.emit(path, img)
                    
                    if self.processed_count >= self.total_files:
                        self.stop() # Auto stop when done
                        self.finished_signal.emit()
                        return
                        
            except Empty:
                break
            except Exception as e:
                print(f"Queue Polling Error: {e}")
                break

    def stop(self):
        """모든 프로세스 강제 종료"""
        self.timer.stop() # Stop polling
        
        self.log_signal.emit("[Manager] Terminating all processes...")
        
        # 워커 프로세스 종료
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        self.processes.clear()
            
        # 큐 정리
        while not self.input_queue.empty():
            try: self.input_queue.get_nowait()
            except: break
            
        self.log_signal.emit("[System] Processing stopped and resources cleaned up.")
