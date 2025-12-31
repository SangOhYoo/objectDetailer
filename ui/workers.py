import os
import cv2
import numpy as np
import torch
import threading
from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PIL import Image

# Local Logic Imports
from core.mask_utils import MaskUtils
from core.sam_wrapper import SamInference

# Diffusers Imports
from diffusers import (
    StableDiffusionInpaintPipeline, 
    StableDiffusionControlNetInpaintPipeline, 
    ControlNetModel
)

# --- GLOBAL SHARED RESOURCES ---
_global_load_lock = threading.Lock() # 모델 로딩 시 충돌 방지용 Lock
_job_queue = Queue() # 파일 처리 대기열 (Producer-Consumer Pattern)

class ProcessingController(QObject):
    """
    메인 윈도우와 통신하며 GPU Worker들을 관리하는 컨트롤러.
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
        if self.processed_count >= self.total_files:
            self.finished_signal.emit()


class GpuWorker(QThread):
    """
    개별 GPU 워커.
    Detection -> Dynamic Denoise -> SAM Masking -> ControlNet Inpaint 파이프라인 수행
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
        self.sam = None         # SAM Wrapper Instance

    def run(self):
        self.log_signal.emit(f"[Worker-{self.worker_id}] Started on {self.device_id}")

        while True:
            try:
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

                # 2. Process Image (Multi-Pass)
                result_image = self.process_pipeline(image)
                
                # 3. Emit Result
                self.result_signal.emit(file_path, result_image)

            except Exception as e:
                self.log_signal.emit(f"[Worker-{self.worker_id}] Error: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                _job_queue.task_done()

        self.log_signal.emit(f"[Worker-{self.worker_id}] Queue empty. Worker Stopping.")
        self.finished_signal.emit()

    def load_sd_model(self, model_path="models/sd_v15_inpaint.safetensors", use_controlnet=False):
        """
        [Load-Move-Cast Pattern] & ControlNet Support
        """
        is_cn_loaded = isinstance(self.pipe, StableDiffusionControlNetInpaintPipeline)
        if self.pipe is not None and is_cn_loaded == use_controlnet:
            return

        with _global_load_lock:
            self.log_signal.emit(f"[Worker-{self.worker_id}] Loading Model (Lock Held)...")
            
            # Clean up
            if self.pipe: del self.pipe
            torch.cuda.empty_cache()

            # 1. Load ControlNet
            controlnet = None
            if use_controlnet:
                # [TODO] 실제 경로로 수정 필요
                cn_path = "lllyasviel/control_v11p_sd15_canny" 
                controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=torch.float16)

            # 2. Load Pipeline (CPU/FP32)
            load_args = {"torch_dtype": torch.float32, "safety_checker": None}
            if controlnet:
                PipelineClass = StableDiffusionControlNetInpaintPipeline
                load_args["controlnet"] = controlnet
            else:
                PipelineClass = StableDiffusionInpaintPipeline

            # [TODO] 실제 로컬 파일 경로 사용 시: from_single_file
            model_id = "runwayml/stable-diffusion-inpainting"
            self.pipe = PipelineClass.from_pretrained(model_id, **load_args)

            # 3. GPU Move & Cast
            self.pipe.to(self.device_id)
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.text_encoder.to(dtype=torch.float16)
            if controlnet:
                self.pipe.controlnet.to(dtype=torch.float16)
            self.pipe.vae.to(dtype=torch.float32) # VAE Safety

            self.log_signal.emit(f"[Worker-{self.worker_id}] Model loaded.")

    def load_sam_model(self):
        """Lazy Load SAM"""
        if self.sam is None:
            with _global_load_lock:
                self.sam = SamInference(device=self.device_id)

    def process_pipeline(self, image):
        result_img = image.copy()
        
        for i, config in enumerate(self.configs):
            if not config['enabled']: continue
            
            pass_name = f"Unit {i+1}"
            # Load appropriate pipeline
            self.load_sd_model(use_controlnet=config['use_controlnet'])
            
            result_img = self.process_pass(result_img, config, pass_name)
        
        return result_img

    def process_pass(self, image, config, pass_name):
        h, w = image.shape[:2]
        
        # [A] Detection (Mock)
        # TODO: Replace with self.detector.detect(image, config['model'])
        # Dummy box: Center crop
        boxes = [[w//4, h//4, w*3//4, h*3//4]] 

        if not boxes: return image

        # SAM Setup
        if config['use_sam']:
            self.load_sam_model()
            self.sam.set_image(image) # Embedding once

        final_img = image.copy()

        # [B] Process Loop
        for box in boxes:
            # 1. Mask Generation
            if config['use_sam'] and self.sam:
                mask = self.sam.predict_mask_from_box(box)
            else:
                mask = MaskUtils.box_to_mask(box, (h, w), padding=0)

            # 2. Mask Refinement
            mask = MaskUtils.refine_mask(mask, dilation=config['dilation'], blur=config['blur'])

            # 3. Dynamic Denoising
            base_denoise = config['denoise']
            final_denoise = self._calc_dynamic_denoise(box, (h, w), base_denoise)
            
            # 4. LoRA Injection (Stub)
            # if config['lora_model'] != "None": load_lora...

            # 5. Inpainting
            final_img = self.run_inpaint_on_mask(final_img, mask, config, final_denoise)
            
            # 6. LoRA Cleanup (Stub)
            # if config['lora_model'] != "None": unload_lora...

        return final_img

    def _calc_dynamic_denoise(self, box, img_shape, base_strength):
        """BMAB Style Dynamic Denoising"""
        x1, y1, x2, y2 = box
        ratio = ((x2 - x1) * (y2 - y1)) / (img_shape[0] * img_shape[1])
        
        adjustment = 0.0
        if ratio < 0.05: adjustment = 0.15   # Very small -> Boost
        elif ratio < 0.10: adjustment = 0.10 # Small -> Boost
        elif ratio < 0.20: adjustment = 0.05 # Medium -> Slight Boost
        
        final = base_strength + adjustment
        return max(0.1, min(final, 0.8))

    def run_inpaint_on_mask(self, image, mask, config, strength):
        # 1. Crop
        padding = config['padding']
        crop_img, (x1, y1, x2, y2) = MaskUtils.crop_image_by_mask(image, mask, context_padding=padding)
        crop_mask = mask[y1:y2, x1:x2]

        if crop_img.size == 0: return image

        # 2. Prep Inputs
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(crop_mask)

        # 3. ControlNet Prep (Canny)
        control_args = {}
        if config['use_controlnet']:
            canny_np = cv2.Canny(crop_img, 100, 200)
            canny_np = np.stack([canny_np]*3, axis=-1)
            pil_control = Image.fromarray(canny_np)
            control_args["control_image"] = pil_control
            control_args["controlnet_conditioning_scale"] = config['cn_weight']

        # 4. Inference
        with torch.inference_mode():
            with torch.autocast(self.device_id.split(':')[0]):
                output = self.pipe(
                    prompt=config['pos_prompt'],
                    negative_prompt=config['neg_prompt'],
                    image=pil_img,
                    mask_image=pil_mask,
                    strength=strength,
                    width=pil_img.width - (pil_img.width % 8),
                    height=pil_img.height - (pil_img.height % 8),
                    **control_args
                ).images[0]

        # 5. Paste
        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        res_np = cv2.resize(res_np, (x2-x1, y2-y1)) # Safety resize
        
        # Simple paste (Alpha blending recommended for production)
        image[y1:y2, x1:x2] = res_np
        
        return image