import os
import cv2
import numpy as np
import torch
import threading
from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PIL import Image

# Core Modules
from core.mask_utils import MaskUtils
from core.detector import ObjectDetector
from core.sam_wrapper import SamInference
from core.config import config_instance as cfg  # Config 싱글톤 임포트

# Diffusers Imports
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL
)

# --- Global Resources ---
_global_load_lock = threading.Lock() # 모델 로딩 충돌 방지
_job_queue = Queue() # 작업 대기열

class ProcessingController(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int) # current, total
    result_signal = pyqtSignal(str, np.ndarray) # filepath, image
    finished_signal = pyqtSignal()

    def __init__(self, file_paths, configs):
        super().__init__()
        self.file_paths = file_paths
        self.configs = configs
        self.workers = []
        self.processed_count = 0
        self.total_files = len(file_paths)

    def start_processing(self):
        # 1. 큐 채우기
        for f in self.file_paths:
            _job_queue.put(f)

        # 2. GPU 감지
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Detected {gpu_count} NVIDIA GPUs.")
        else:
            gpu_count = 1
            self.log_signal.emit("[System] No GPU detected. Using CPU (Slow).")

        # 3. 워커 생성
        for i in range(gpu_count):
            dev_id = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            worker = GpuWorker(dev_id, self.configs, worker_id=i)
            
            worker.log_signal.connect(self.relay_log)
            worker.result_signal.connect(self.handle_result)
            worker.finished_signal.connect(self.check_finished)
            
            self.workers.append(worker)
            worker.start()

    def relay_log(self, msg):
        self.log_signal.emit(msg)

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
        
        self.pipe = None
        self.sam = None
        
        # [Config] Detector 초기화 (ADetailer 모델 경로 주입)
        # 1순위: config의 adetailer 경로, 없으면 sam 경로 등 활용 가능. 여기선 sam 경로 예시.
        model_dir = cfg.get_path('sam') 
        self.detector = ObjectDetector(device=device_id, model_dir=model_dir)

    def run(self):
        self.log_signal.emit(f"[Worker-{self.worker_id}] Initialized on {self.device_id}")

        while not _job_queue.empty():
            try:
                fpath = _job_queue.get_nowait()
            except:
                break

            self.log_signal.emit(f"[Worker-{self.worker_id}] Processing: {os.path.basename(fpath)}")
            try:
                # [FIX] 한글 경로 이미지 로딩 호환성 수정
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    self.log_signal.emit(f"[Worker-{self.worker_id}] Error: Failed to load image.")
                    continue
                
                result_img = self.process_image(img)
                self.result_signal.emit(fpath, result_img)
                
            except Exception as e:
                self.log_signal.emit(f"[Worker-{self.worker_id}] Critical Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                _job_queue.task_done()
        
        self.log_signal.emit(f"[Worker-{self.worker_id}] Task finished.")
        self.finished_signal.emit()

    def process_image(self, image):
        result_img = image.copy()
        
        for i, config in enumerate(self.configs):
            if not config['enabled']: continue
            
            unit_name = f"Unit {i+1}"
            self.log_signal.emit(f"  > [Worker-{self.worker_id}] Running {unit_name}...")
            
            # 파이프라인 및 LoRA 준비
            self.load_sd_model(use_controlnet=config['use_controlnet'])
            self.manage_lora(config, action="load")

            try:
                result_img = self.process_pass(result_img, config)
            finally:
                self.manage_lora(config, action="unload")
            
        return result_img

    def process_pass(self, image, config):
        h, w = image.shape[:2]
        img_area = h * w
        
        # 1. 탐지
        detections = self.detector.detect(image, config['model'], config['conf'])
        if not detections: return image

        # 2. 면적순 정렬
        detections.sort(key=lambda d: (d['box'][2]-d['box'][0]) * (d['box'][3]-d['box'][1]), reverse=True)

        # 3. SAM 준비 (필요 시)
        if config['use_sam']:
            if self.sam is None:
                with _global_load_lock:
                    # [Config] SAM 모델 경로 로드
                    sam_file = cfg.get_path('sam', 'sam_file')
                    self.sam = SamInference(checkpoint=sam_file, device=self.device_id)
            self.sam.set_image(image)

        final_img = image.copy()

        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            
            # [FIX] 면적 필터링 (Area Filtering)
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / img_area
            if ratio < config['min_area'] or ratio > config['max_area']:
                continue

            # 마스크 결정 (SAM > YOLO Seg > Box)
            yolo_mask = det['mask']
            
            if config['use_sam'] and self.sam:
                mask = self.sam.predict_mask_from_box(box)
            elif yolo_mask is not None:
                mask = yolo_mask
            else:
                mask = MaskUtils.box_to_mask(box, (h, w), padding=0)

            # 마스크 정제
            mask = MaskUtils.refine_mask(mask, dilation=config['dilation'], blur=config['blur'])
            
            # BMAB Dynamic Denoising
            final_denoise = self._calc_dynamic_denoise(box, (h, w), config['denoise'])
            
            # 인페인팅 수행
            final_img = self.run_inpaint(final_img, mask, config, final_denoise)

        return final_img

    def run_inpaint(self, image, mask, config, strength):
        padding = config['padding']
        crop_img, (x1, y1, x2, y2) = MaskUtils.crop_image_by_mask(image, mask, context_padding=padding)
        crop_mask = mask[y1:y2, x1:x2]
        
        if crop_img.size == 0: return image

        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(crop_mask)

        # ControlNet 전처리 (Canny)
        control_args = {}
        if config['use_controlnet']:
            canny = cv2.Canny(crop_img, 100, 200)
            canny = np.stack([canny] * 3, axis=-1)
            pil_control = Image.fromarray(canny)
            control_args["control_image"] = pil_control
            control_args["controlnet_conditioning_scale"] = config['cn_weight']

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

        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        res_np = cv2.resize(res_np, (x2 - x1, y2 - y1))
        image[y1:y2, x1:x2] = res_np
        
        return image

    def _calc_dynamic_denoise(self, box, img_shape, base_strength):
        x1, y1, x2, y2 = box
        ratio = ((x2 - x1) * (y2 - y1)) / (img_shape[0] * img_shape[1])
        adjustment = 0.0
        if ratio < 0.05: adjustment = 0.15
        elif ratio < 0.10: adjustment = 0.10
        elif ratio < 0.20: adjustment = 0.05
        return max(0.1, min(base_strength + adjustment, 0.8))

    def load_sd_model(self, use_controlnet=False):
        is_cn_loaded = isinstance(self.pipe, StableDiffusionControlNetInpaintPipeline)
        if self.pipe is not None and is_cn_loaded == use_controlnet:
            return

        with _global_load_lock:
            if self.pipe: del self.pipe
            torch.cuda.empty_cache()

            # [Config] 모델 경로 로드
            ckpt_path = cfg.get_path('checkpoint', 'checkpoint_file')
            vae_path = cfg.get_path('vae', 'vae_file')
            
            # ControlNet 로드
            controlnet = None
            if use_controlnet:
                cn_path = cfg.get_path('controlnet', 'controlnet_tile')
                if cn_path and os.path.exists(cn_path):
                    controlnet = ControlNetModel.from_single_file(cn_path, torch_dtype=torch.float16)
                else:
                    self.log_signal.emit(f"    [Warning] ControlNet not found at {cn_path}")

            # VAE 로드
            vae = None
            if vae_path and os.path.exists(vae_path):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)

            load_args = {
                "torch_dtype": torch.float32, 
                "safety_checker": None,
                "use_safetensors": True
            }
            if vae: load_args["vae"] = vae
            
            if controlnet:
                PipelineClass = StableDiffusionControlNetInpaintPipeline
                load_args["controlnet"] = controlnet
            else:
                PipelineClass = StableDiffusionInpaintPipeline

            if not os.path.exists(ckpt_path):
                self.log_signal.emit(f"[Error] Checkpoint not found: {ckpt_path}")
                return

            self.pipe = PipelineClass.from_single_file(ckpt_path, **load_args)

            self.pipe.to(self.device_id)
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.text_encoder.to(dtype=torch.float16)
            if controlnet: self.pipe.controlnet.to(dtype=torch.float16)
            if self.pipe.vae: self.pipe.vae.to(dtype=torch.float32)

    def manage_lora(self, config, action="load"):
        lora_name = config.get('lora_model', 'None')
        if lora_name == "None": return

        try:
            if action == "load":
                # [Config] LoRA 경로 로드
                lora_base = cfg.get_path('lora')
                if not lora_base: return
                
                lora_path = os.path.join(lora_base, lora_name)
                
                if not os.path.exists(lora_path):
                    self.log_signal.emit(f"    [Warning] LoRA not found: {lora_path}")
                    return

                self.pipe.load_lora_weights(lora_path, adapter_name="default")
                self.pipe.fuse_lora(lora_scale=config['lora_scale'])
                self.log_signal.emit(f"    [LoRA] Injected: {lora_name}")

            elif action == "unload":
                self.pipe.unfuse_lora()
                self.pipe.unload_lora_weights()
                
        except Exception as e:
            self.log_signal.emit(f"    [LoRA Error] {str(e)}")