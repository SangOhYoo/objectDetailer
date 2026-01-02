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

from core.pipeline import ImageProcessor
from core.config import config_instance as cfg
from core.metadata import save_image_with_metadata

def worker_process(device_id, input_queue, output_queue, configs):
    """
    별도의 프로세스에서 실행되는 워커 함수.
    QThread가 아닌 multiprocessing.Process로 실행됨.
    """
    worker_id = device_id
    try:
        # [MP] 프로세스 별로 ImageProcessor 초기화 (독립 메모리)
        # 로그는 큐를 통해 메인 프로세스로 전달
        def log_wrapper(msg):
            output_queue.put(("log", f"[{worker_id}] {msg}"))
            
        def preview_wrapper(img):
            # Numpy로 변환하여 전송 (Pickle 가능하도록)
            if img is not None and not isinstance(img, np.ndarray):
                try:
                    img = np.array(img)
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except:
                    pass
            output_queue.put(("preview", img))

        processor = ImageProcessor(device_id, log_callback=log_wrapper, preview_callback=preview_wrapper)
        log_wrapper(f"Initialized on {device_id} (PID: {os.getpid()})")

        while True:
            try:
                # 큐에서 작업 가져오기 (타임아웃 1초)
                task = input_queue.get(timeout=1.0)
            except Empty:
                # 타임아웃 발생 시 다시 루프 (종료 시그널 확인 등 필요 시 확장 가능)
                continue
                
            if task is None: # 종료 시그널
                break
                
            fpath = task
            output_queue.put(("started", os.path.basename(fpath)))
            log_wrapper(f"Processing: {os.path.basename(fpath)}")
            
            try:
                # 이미지 로드
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    log_wrapper("Error: Load failed.")
                    output_queue.put(("result", (fpath, None)))
                    continue
                
                # 프리뷰 전송
                preview_wrapper(img)
                
                # 처리 실행
                result_img = processor.process(img, configs)
                
                # 결과 저장 로직 (프로세스 내부에서 저장까지 완료)
                # Config가 Pickling되어 넘어왔으므로 여기서 cfg 인스턴스 접근 시 주의 필요
                # 하지만 save_image_with_metadata는 경로만 알면 됨. output_path는 다시 읽거나 전달받아야 함.
                # 편의상 cfg를 여기서 다시 로드하거나 safe한 경로 사용
                # 여기서는 메인 프로세스에서 전달된 configs를 사용
                
                # NOTE: cfg.get()은 파일에서 읽는 것이므로 MP에서도 동작할 수 있으나, 
                # 동기화 문제가 있을 수 있으니 저장 로직을 간소화
                
                output_dir = "outputs" # Default
                # cfg 인스턴스는 각 프로세스에서 독립적이므로 다시 로드하지 않으면 기본값일 수 있음.
                # 하지만 workers.py import 시점에 초기화되므로 config.yaml을 읽었을 것임.
                if cfg.get('system', 'output_path'):
                    output_dir = cfg.get('system', 'output_path')
                
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(fpath)
                save_path = os.path.join(output_dir, filename)
                
                # [Optimization] 비동기 저장 (Async Save)
                # 디스크 쓰기(I/O)가 GPU 추론을 막지 않도록 별도 스레드풀에서 처리
                # Config Wrapper 함수 정의 (Pickle 문제 방지 위해 내부 정의 대신 Top-level 선호하지만 여기선 closure 사용 안함)
                
                # 메타데이터 저장용 래퍼
                active_conf = next((c for c in configs if c.get('enabled')), {})
                class ConfigWrapper:
                    def __init__(self, d): self.d = d
                    def to_adetailer_json(self): return self.d
                
                # 저장 함수 (별도 스레드에서 실행)
                def save_task(img_to_save, src_path, dst_path, conf_wrapper):
                    try:
                        if not save_image_with_metadata(img_to_save, src_path, dst_path, conf_wrapper):
                            # 메인 큐에 로그를 보낼 수 없으므로(Process Queue는 피클링 필요), 
                            # 여기서 에러가 나면 콘솔에만 찍거나 무시해야 함. 
                            # 하지만 log_wrapper는 메인 루프 변수라 접근 불가.
                            # 해결책: save_task에 log_queue도 전달하거나, 그냥 에러만 출력.
                            print(f"[AsyncSave] Warning: Failed to save {os.path.basename(src_path)}")
                    except Exception as e:
                        print(f"[AsyncSave] Error saving {os.path.basename(src_path)}: {e}")

                # 스레드풀이 없으면 생성 (Process 당 1개)
                if not hasattr(worker_process, "executor"):
                     from concurrent.futures import ThreadPoolExecutor
                     worker_process.executor = ThreadPoolExecutor(max_workers=1)

                worker_process.executor.submit(save_task, result_img, fpath, save_path, ConfigWrapper(active_conf))

                # 결과 이미지(Numpy)를 큐로 전송하면 비용이 큼. 
                # 메인 UI 표시용으로는 썸네일이나 경로만 줘도 되지만, 
                # 현재 구조상 이미지를 통째로 넘겨야 비교 뷰어가 갱신됨.
                output_queue.put(("result", (fpath, result_img)))

            except Exception as e:
                log_wrapper(f"Critical Error: {e}")
                traceback.print_exc()
                output_queue.put(("result", (fpath, None)))
            finally:
                # 메모리 정리
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"Worker Process Died: {e}")
        traceback.print_exc()

class ResultListener(QThread):
    """
    멀티프로세싱 큐(Output Queue)를 모니터링하여 메인 스레드에 시그널을 보내는 리스너.
    """
    log_signal = pyqtSignal(str)
    preview_signal = pyqtSignal(np.ndarray)
    file_started_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str, object)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int, int) # (processed, total)

    def __init__(self, output_queue, total_files):
        super().__init__()
        self.output_queue = output_queue
        self.total_files = total_files
        self.processed_count = 0
        self.is_running = True

    def run(self):
        while self.is_running:
            try:
                # 0.1초 대기하며 폴링 (UI 블로킹 방지)
                msg_type, data = self.output_queue.get(timeout=0.1)
                
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
                        self.finished_signal.emit()
                        # 모든 작업 완료 시 루프 종료 (하지만 큐에 남은 메시지가 있을 수 있으므로 continue)
                        # 여기서는 종료하지 않고 계속 리슨하다가 컨트롤러가 종료시킴
            except Empty:
                continue
            except Exception as e:
                print(f"Listener Error: {e}")
                break

    def stop(self):
        self.is_running = False

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
        self.listener = None

    def start_processing(self):
        # 작업 큐 채우기
        for f in self.file_paths:
            self.input_queue.put(f)

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Detected {gpu_count} NVIDIA GPUs. Starting Multiprocessing...")
        else:
            gpu_count = 1
            self.log_signal.emit("[System] No GPU detected. Using CPU via Multiprocessing.")

        # 리스너 시작
        self.listener = ResultListener(self.output_queue, len(self.file_paths))
        self.listener.log_signal.connect(self.log_signal.emit)
        self.listener.preview_signal.connect(self.preview_signal.emit)
        self.listener.file_started_signal.connect(self.file_started_signal.emit)
        self.listener.result_signal.connect(self.result_signal.emit) # handle_result to main window
        self.listener.progress_signal.connect(self.progress_signal.emit)
        self.listener.finished_signal.connect(self.finished_signal.emit)
        self.listener.start()

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

    def stop(self):
        """모든 프로세스 강제 종료"""
        self.log_signal.emit("[Manager] Terminating all processes...")
        
        # 워커 프로세스 종료
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        self.processes.clear()
        
        # 리스너 종료
        if self.listener:
            self.listener.stop()
            self.listener.wait()
            self.listener = None
            
        # 큐 정리 (선택적)
        while not self.input_queue.empty():
            try: self.input_queue.get_nowait()
            except: break
            
        self.log_signal.emit("[System] Processing stopped and resources cleaned up.")