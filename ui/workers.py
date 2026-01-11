import os
import cv2
import numpy as np
import traceback
import warnings
import multiprocessing as mp
import time
from queue import Empty
import gc
from PyQt6.QtCore import pyqtSignal, QObject, QTimer

# [Fix] Suppress annoying FutureWarnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

def worker_process(device_id, input_queue, output_queue, configs):
    """
    Worker function running in a separate process.
    """
    worker_id = device_id
    
    # 1. Ultimate Isolation Strategy
    if "cuda" in device_id:
        try:
            target_idx = int(device_id.split(':')[1])
            # Set environment variable at the very start of the process
            os.environ["CUDA_VISIBLE_DEVICES"] = str(target_idx)
        except:
            pass
            
    vis_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
    
    # 2. Late Imports (Required for isolation on Windows spawn)
    import torch
    import torchvision
    if not hasattr(torchvision.transforms, "functional_tensor"):
        import torchvision.transforms.functional as F
        import sys as sys_mod
        sys_mod.modules["torchvision.transforms.functional_tensor"] = F
    
    from core.pipeline import ImageProcessor
    from core.config import config_instance as cfg
    from core.metadata import save_image_with_metadata
    
    internal_device_id = "cuda:0" if "cuda" in device_id else "cpu"

    print(f"[Process Start] PID: {os.getpid()}, Req: {device_id}, Env: {vis_dev} -> Using: {internal_device_id}")

    try:
        def log_wrapper(msg):
            output_queue.put(("log", f"[{worker_id}] {msg}"))
            
        def preview_wrapper(img):
            if img is not None and not isinstance(img, np.ndarray):
                try:
                    img = np.array(img)
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except:
                    pass
            output_queue.put(("preview", img))

        if "cuda" in internal_device_id:
            try:
                avail = torch.cuda.is_available()
                cnt = torch.cuda.device_count()
                
                if avail and cnt > 0:
                    props = torch.cuda.get_device_properties(0)
                    total_vram = props.total_memory / (1024**3)
                    msg = f"GPU Verified: {props.name} ({total_vram:.1f}GB VRAM) | VisibleCount={cnt}"
                    log_wrapper(msg)
                    print(f"[Worker {worker_id}] {msg}")
                    
                    torch.cuda.set_device(0)
                    t = torch.tensor([1.0], device="cuda:0")
                    del t
                else:
                    raise RuntimeError("CUDA not detected in isolated worker process.")
            except Exception as e:
                err_msg = f"Error: GPU {device_id} init failed ({e}). Falling back to CPU."
                log_wrapper(err_msg)
                internal_device_id = "cpu"

        processor = ImageProcessor(internal_device_id, log_callback=log_wrapper, preview_callback=preview_wrapper)
        log_wrapper(f"ImageProcessor Initialized on {internal_device_id}")

        while True:
            try:
                # [Fix] Reduced timeout from 1.0 -> 0.1 for faster shutdown/responsiveness
                task = input_queue.get(timeout=0.1)
            except Empty:
                continue
                
            if task is None: # Exit signal
                break
                
            if isinstance(task, tuple):
                fpath, angle = task
            else:
                fpath, angle = task, 0
                
            output_queue.put(("started", os.path.basename(fpath)))
            
            try:
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    log_wrapper("Error: Image load failed.")
                    output_queue.put(("result", (fpath, None)))
                    continue
                
                if angle != 0:
                    if angle == 90: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180: img = cv2.rotate(img, cv2.ROTATE_180)
                    elif angle == 270: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                preview_wrapper(img)
                
                result_img = processor.process(img, configs)
                
                if angle != 0 and result_img is not None:
                     if angle == 90: result_img = cv2.rotate(result_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                     elif angle == 180: result_img = cv2.rotate(result_img, cv2.ROTATE_180)
                     elif angle == 270: result_img = cv2.rotate(result_img, cv2.ROTATE_90_CLOCKWISE)

                output_dir = cfg.get('system', 'output_path') or "outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(fpath)
                save_path = os.path.join(output_dir, filename)
                
                active_conf = next((c for c in configs if c.get('enabled')), {})
                class ConfigWrapper:
                    def __init__(self, d): self.d = d
                    def to_adetailer_json(self): return self.d
                
                save_image_with_metadata(result_img, fpath, save_path, ConfigWrapper(active_conf))

                output_queue.put(("result", (fpath, result_img)))

            except Exception as e:

                log_wrapper(f"Critical Worker Error: {e}")
                traceback.print_exc()
                output_queue.put(("result", (fpath, None)))

        # [Final Cleanup] Offload everything when worker loop ends
        try:
            processor.detector.offload_models()
            if processor.sam: processor.sam.unload_model()
            processor.face_restorer.unload_model()
            processor.model_manager.unload_model()
            gc.collect()
            if "cuda" in internal_device_id:
                torch.cuda.empty_cache()
        except:
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
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_queue)
        self.total_files = len(file_paths)
        self.processed_count = 0

    def start_processing(self, max_workers=1):
        import torch
        
        for f in self.file_paths:
            self.input_queue.put(f)

        if torch.cuda.is_available():
            real_gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Multi-GPU Controller Loaded: {real_gpu_count} GPUs available.")
            
            orig_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            
            try:
                for i in range(max_workers):
                    gpu_idx = i % real_gpu_count
                    dev_id = f"cuda:{gpu_idx}"
                    
                    # Parent-Level Isolation
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                    
                    p = mp.Process(
                        target=worker_process, 
                        args=(dev_id, self.input_queue, self.output_queue, self.configs),
                        daemon=True
                    )
                    p.start()
                    self.processes.append(p)
                    self.log_signal.emit(f"[Manager] Spawned Worker {i+1} for Physical GPU {gpu_idx} (PID: {p.pid})")
                    time.sleep(0.1) # Stabilization sleep
            finally:
                if orig_vis is None:
                    if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = orig_vis
                
        else:
            self.log_signal.emit(f"[System] CPU Parallel Processing Started.")
            for i in range(max_workers):
                p = mp.Process(
                    target=worker_process, 
                    args=("cpu", self.input_queue, self.output_queue, self.configs),
                    daemon=True
                )
                p.start()
                self.processes.append(p)
                self.log_signal.emit(f"[Manager] Started CPU Worker {i+1} (PID: {p.pid})")
            
        self.timer.start(50)

    def check_queue(self):
        for _ in range(20): 
            try:
                msg_type, data = self.output_queue.get_nowait()
                if msg_type == "log": self.log_signal.emit(data)
                elif msg_type == "preview": self.preview_signal.emit(data)
                elif msg_type == "started": self.file_started_signal.emit(data)
                elif msg_type == "result":
                    path, img = data
                    self.processed_count += 1
                    self.progress_signal.emit(self.processed_count, self.total_files)
                    self.result_signal.emit(path, img)
                    if self.processed_count >= self.total_files:
                        self.stop()
                        self.finished_signal.emit()
                        return
            except Empty: break
            except Exception as e:
                print(f"Queue Polling Error: {e}")
                break

    def stop(self):
        """
        [Fix] Faster Shutdown: Avoid sequential long join timeouts.
        """
        self.timer.stop()
        
        active_workers = [p for p in self.processes if p.is_alive()]
        if not active_workers:
            self.processes.clear()
            return

        self.log_signal.emit("[Manager] Stopping workers...")
        
        # 1. Signal ALL workers to exit first (Non-blocking)
        for _ in range(len(active_workers)):
            try: 
                self.input_queue.put_nowait(None)
            except: 
                pass
        
        # 2. Join with small timeout
        for p in active_workers:
            p.join(timeout=0.3)
            
        # 3. Force terminate any that are still stuck
        for p in active_workers:
            if p.is_alive():
                try:
                    self.log_signal.emit(f"[Manager] Process {p.pid} still alive. Terminating forcefully...")
                    p.terminate()
                    p.join(timeout=0.2)
                except:
                    pass
        
        self.processes.clear()
        try:
            # [Fix] Cancel join_thread to avoid hanging/delay on Windows
            self.input_queue.close()
            self.input_queue.cancel_join_thread()
            self.output_queue.close()
            self.output_queue.cancel_join_thread()
        except:
            pass
        
        # [New] Diagnostic VRAM Check in Parent (after cleanup)
        import torch
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
             gc.collect()
             # Log final usage (approximate since it's parent process)
             vram_rem = torch.cuda.memory_reserved(0) / (1024**2)
             self.log_signal.emit(f"[System] Multi-GPU Cleanup Complete. Reserved VRAM (GUI Process): {vram_rem:.1f}MB")

        self.log_signal.emit("[System] Processing ended.")
