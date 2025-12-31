import os
import cv2
import numpy as np
import torch
import threading
from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PIL import Image

# Import Core Logic (Assuming these exist in your project structure)
from core.mask_utils import MaskUtils
# from core.detector import ObjectDetector # Placeholder for actual detector

# Diffusers Imports for Stable Diffusion & ControlNet
from diffusers import (
    StableDiffusionInpaintPipeline, 
    StableDiffusionControlNetInpaintPipeline, 
    ControlNetModel,
    AutoencoderKL
)

# --- GLOBAL SHARED RESOURCES ---
_global_load_lock = threading.Lock() # 모델 로딩 시 충돌 방지용 Lock
_job_queue = Queue() # 파일 처리 대기열 (Producer-Consumer Pattern)

class ProcessingController(QObject):
    """
    메인 윈도우와 통신하며 GPU Worker들을 관리하는 컨트롤러.
    파일을 Queue에 넣고, 가용한 GPU 개수만큼 Worker를 생성합니다.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int) # current, total
    result_signal = pyqtSignal(str, np.ndarray) # filepath, image
    finished_signal = pyqtSignal()

    def __init__(self, file_paths, configs):
        super().__init__()
        self.file_paths = file_paths
        self.configs = configs # List of config dicts from UI Tabs
        self.workers = []
        self.processed_count = 0
        self.total_files = len(file_paths)

    def start_processing(self):
        # 1. Fill the Queue
        for fpath in self.file_paths:
            _job_queue.put(fpath)

        # 2. Detect GPUs
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Detected {gpu_count} NVIDIA GPUs.")
        else:
            gpu_count = 1 # CPU Fallback
            self.log_signal.emit("[System] No GPU detected. Using CPU (Slow).")

        # 3. Spawn Workers
        for i in range(gpu_count):
            device_id = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            worker = GpuWorker(device_id, self.configs, worker_id=i)
            
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
        # 단순히 카운트로 체크하지 않고, 모든 워커가 종료되었는지 확인하는 것이 안전하나
        # 여기서는 처리된 파일 수로 1차 판단
        if self.processed_count >= self.total_files:
            self.finished_signal.emit()


class GpuWorker(QThread):
    """
    개별 GPU에 할당되어 독립적으로 작업을 수행하는 워커.
    자신만의 Pipeline, Detector, Memory Context를 가집니다.
    """
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str, np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, device_id, configs, worker_id):
        super().__init__()
        self.device_id = device_id
        self.configs = configs
        self.worker_id = worker_id
        
        # Models
        self.pipe = None        # Stable Diffusion Pipeline
        self.detector = None    # YOLO/MediaPipe
        self.sam = None         # Segment Anything Model

    def run(self):
        self.log_signal.emit(f"[Worker-{self.worker_id}] Started on {self.device_id}")

        while True:
            try:
                # Non-blocking get with timeout is safer to avoid deadlocks
                if _job_queue.empty():
                    break
                
                file_path = _job_queue.get_nowait()
            except:
                break

            self.log_signal.emit(f"[Worker-{self.worker_id}] Processing: {os.path.basename(file_path)}")
            
            try:
                # 1. Load Image
                image = cv2.imread(file_path)
                if image is None:
                    self.log_signal.emit(f"[Worker-{self.worker_id}] Failed to load image.")
                    continue

                # 2. Process Image (Multi-Pass Pipeline)
                result_image = self.process_pipeline(image)
                
                # 3. Emit Result
                self.result_signal.emit(file_path, result_image)

            except Exception as e:
                self.log_signal.emit(f"[Worker-{self.worker_id}] Error: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                _job_queue.task_done()

        self.log_signal.emit(f"[Worker-{self.worker_id}] Queue empty. Stopping.")
        self.finished_signal.emit()

    def load_sd_model(self, model_path="models/sd_v15_inpaint.safetensors", use_controlnet=False):
        """
        [Load-Move-Cast Pattern]
        Global Lock을 사용하여 스레드 안전하게 모델을 로드하고 FP16으로 변환합니다.
        BMAB: ControlNet 사용 여부에 따라 파이프라인을 교체합니다.
        """
        # 현재 파이프라인이 요구사항과 일치하면 재사용
        is_cn_loaded = isinstance(self.pipe, StableDiffusionControlNetInpaintPipeline)
        if self.pipe is not None and is_cn_loaded == use_controlnet:
            return

        with _global_load_lock:
            self.log_signal.emit(f"[Worker-{self.worker_id}] Loading/Switching Model (Lock Held)...")
            
            # Clean up memory
            if self.pipe:
                del self.pipe
            torch.cuda.empty_cache()

            # 1. Load ControlNet (Optional)
            controlnet = None
            if use_controlnet:
                # 실제로는 로컬 경로를 사용해야 함
                cn_path = "lllyasviel/control_v11p_sd15_canny" 
                controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=torch.float16)

            # 2. Load Pipeline (CPU -> FP32)
            load_args = {
                "torch_dtype": torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False
            }

            if controlnet:
                PipelineClass = StableDiffusionControlNetInpaintPipeline
                load_args["controlnet"] = controlnet
            else:
                PipelineClass = StableDiffusionInpaintPipeline

            # Load from file (Mock path provided, replace with actual)
            # self.pipe = PipelineClass.from_single_file(model_path, **load_args)
            
            # [DEV MODE] 로컬 모델 파일이 없을 경우를 대비해 from_pretrained 사용 (실제 배포시 수정)
            model_id = "runwayml/stable-diffusion-inpainting"
            self.pipe = PipelineClass.from_pretrained(model_id, **load_args)

            # 3. Move to GPU
            self.pipe.to(self.device_id)

            # 4. Cast to FP16 (Optimization)
            # VAE는 정밀도를 위해 FP32 유지 권장
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.text_encoder.to(dtype=torch.float16)
            if controlnet:
                self.pipe.controlnet.to(dtype=torch.float16)
            self.pipe.vae.to(dtype=torch.float32)

            self.log_signal.emit(f"[Worker-{self.worker_id}] Model loaded successfully.")

    def process_pipeline(self, image):
        """
        순차적 패스 처리 (Pass 1 -> Pass 2 -> Pass 3)
        """
        result_img = image.copy()
        
        for i, config in enumerate(self.configs):
            if not config['enabled']:
                continue
            
            pass_name = f"Unit {i+1}"
            self.log_signal.emit(f"[Worker-{self.worker_id}] Running {pass_name} ({config['model']})")
            
            try:
                # Lazy Load Model with correct configuration
                self.load_sd_model(use_controlnet=config['use_controlnet'])
                
                result_img = self.process_pass(result_img, config)
            except Exception as e:
                self.log_signal.emit(f"[Worker-{self.worker_id}] Failed {pass_name}: {e}")
        
        return result_img

    def process_pass(self, image, config):
        """
        단일 패스 로직: Detection -> Masking -> Inpainting (w/ BMAB)
        """
        # A. Detection (Mocking actual detection for structure)
        # boxes = self.detector.detect(image, config['model'], config['conf'])
        h, w = image.shape[:2]
        # Dummy box for testing
        boxes = [[w//3, h//3, w*2//3, h*2//3]] 

        if not boxes:
            return image

        final_img = image.copy()

        # B. Process each detected object
        for box in boxes:
            # 1. Create Mask (SAM or Box)
            # mask = self.sam.predict(image, box)
            mask = MaskUtils.box_to_mask(box, (h, w), padding=0)
            
            # 2. Refine Mask (ADetailer Logic)
            mask = MaskUtils.refine_mask(
                mask, 
                dilation=config['dilation'], 
                blur=config['blur']
            )

            # 3. LoRA Injection (BMAB)
            if config['lora_model'] != "None":
                # self.pipe.load_lora_weights(config['lora_model'])
                # self.pipe.fuse_lora(lora_scale=config['lora_scale'])
                pass # 실제 파일 경로 필요

            # 4. Run Inpaint
            final_img = self.run_inpaint_on_mask(final_img, mask, config)

            # 5. LoRA Cleanup
            if config['lora_model'] != "None":
                # self.pipe.unfuse_lora()
                # self.pipe.unload_lora_weights()
                pass

        return final_img

    def run_inpaint_on_mask(self, image, mask, config):
        """
        Crop -> Inpaint -> Paste
        """
        # 1. Crop
        padding = config['padding']
        crop_img, (x1, y1, x2, y2) = MaskUtils.crop_image_by_mask(image, mask, context_padding=padding)
        crop_mask = mask[y1:y2, x1:x2]

        if crop_img.size == 0: return image

        # 2. Prepare Inputs
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(crop_mask)

        # 3. ControlNet Prep (BMAB Canny)
        control_args = {}
        if config['use_controlnet']:
            # Canny Edge Detection on Cropped Image
            canny_np = cv2.Canny(crop_img, 100, 200)
            canny_np = np.stack([canny_np]*3, axis=-1)
            pil_control = Image.fromarray(canny_np)
            
            control_args["control_image"] = pil_control
            control_args["controlnet_conditioning_scale"] = config['cn_weight']

        # 4. Inference
        # Autocast & Inference Mode for safety and speed
        with torch.inference_mode():
            with torch.autocast(self.device_id.split(':')[0]): # "cuda"
                output = self.pipe(
                    prompt=config['pos_prompt'],
                    negative_prompt=config['neg_prompt'],
                    image=pil_img,
                    mask_image=pil_mask,
                    strength=config['denoise'],
                    width=pil_img.width - (pil_img.width % 8), # SD requires div/8
                    height=pil_img.height - (pil_img.height % 8),
                    **control_args
                ).images[0]

        # 5. Paste Back
        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        
        # Resize back to original crop size (if rounded during inference)
        res_np = cv2.resize(res_np, (x2-x1, y2-y1))
        
        # Simple paste (Production should use alpha blending)
        image[y1:y2, x1:x2] = res_np
        
        return image