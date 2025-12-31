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
from core.config import config_instance as cfg

# Diffusers Imports
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)

_global_load_lock = threading.Lock()
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
        self.workers = []
        self.processed_count = 0
        self.total_files = len(file_paths)

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
        
        model_dir = cfg.get_path('sam') 
        self.detector = ObjectDetector(device=device_id, model_dir=model_dir)

    def run(self):
        self.log_signal.emit(f"[Worker-{self.worker_id}] Initialized on {self.device_id}")

        while not _job_queue.empty():
            try:
                fpath = _job_queue.get_nowait()
            except: break

            self.log_signal.emit(f"[Worker-{self.worker_id}] Processing: {os.path.basename(fpath)}")
            try:
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    self.log_signal.emit(f"[Worker-{self.worker_id}] Error: Load failed.")
                    continue
                
                result_img = self.process_image(img)
                self.result_signal.emit(fpath, result_img)
            except Exception as e:
                self.log_signal.emit(f"[Worker-{self.worker_id}] Error: {e}")
                import traceback; traceback.print_exc()
            finally:
                _job_queue.task_done()
        
        self.log_signal.emit(f"[Worker-{self.worker_id}] Finished.")
        self.finished_signal.emit()

    def process_image(self, image):
        result_img = image.copy()
        
        for i, config in enumerate(self.configs):
            if not config['enabled']: continue
            
            unit_name = f"Unit {i+1}"
            self.log_signal.emit(f"  > [Worker-{self.worker_id}] Running {unit_name}...")
            
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
        
        detections = self.detector.detect(image, config['model'], config['conf'])
        if not detections: return image

        detections.sort(key=lambda d: (d['box'][2]-d['box'][0]) * (d['box'][3]-d['box'][1]), reverse=True)

        if config['use_sam']:
            if self.sam is None:
                with _global_load_lock:
                    sam_file = cfg.get_path('sam', 'sam_file')
                    self.sam = SamInference(checkpoint=sam_file, device=self.device_id)
            self.sam.set_image(image)

        final_img = image.copy()

        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / img_area
            if ratio < config['min_area'] or ratio > config['max_area']:
                continue

            yolo_mask = det['mask']
            
            if config['use_sam'] and self.sam:
                mask = self.sam.predict_mask_from_box(box)
            elif yolo_mask is not None:
                mask = yolo_mask
            else:
                mask = MaskUtils.box_to_mask(box, (h, w), padding=0)

            mask = MaskUtils.refine_mask(mask, dilation=config['dilation'], blur=config['blur'])
            final_denoise = self._calc_dynamic_denoise(box, (h, w), config['denoise'])
            final_img = self.run_inpaint(final_img, mask, config, final_denoise)

        return final_img

    def _apply_scheduler(self, sampler_name):
        if self.pipe is None: return
        config = self.pipe.scheduler.config
        
        if "Euler a" in sampler_name:
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        elif "Euler" in sampler_name:
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(config)
        elif "DPM++ 2M" in sampler_name:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
        elif "DDIM" in sampler_name:
            self.pipe.scheduler = DDIMScheduler.from_config(config)

    def run_inpaint(self, image, mask, config, strength):
        padding = config['padding']
        crop_img, (x1, y1, x2, y2) = MaskUtils.crop_image_by_mask(image, mask, context_padding=padding)
        crop_mask = mask[y1:y2, x1:x2]
        
        if crop_img.size == 0: return image

        # Upscale Logic (High Quality Fix)
        h_orig, w_orig = crop_img.shape[:2]
        target_res = 512
        scale_factor = target_res / max(h_orig, w_orig)
        
        if max(h_orig, w_orig) < target_res:
            new_w = int(w_orig * scale_factor)
            new_h = int(h_orig * scale_factor)
            new_w -= new_w % 8
            new_h -= new_h % 8
            proc_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            proc_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            new_w = w_orig - (w_orig % 8)
            new_h = h_orig - (h_orig % 8)
            proc_img = crop_img[:new_h, :new_w]
            proc_mask = crop_mask[:new_h, :new_w]

        pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(proc_mask)

        control_args = {}
        if config['use_controlnet']:
            canny = cv2.Canny(proc_img, 100, 200)
            canny = np.stack([canny] * 3, axis=-1)
            pil_control = Image.fromarray(canny)
            control_args["control_image"] = pil_control
            control_args["controlnet_conditioning_scale"] = config['cn_weight']

        # Apply Sampler & Seed
        self._apply_scheduler(config.get('sampler', 'Euler a'))
        seed = config.get('seed', -1)
        generator = torch.Generator(device=self.device_id)
        if seed != -1:
            generator.manual_seed(seed)

        with torch.inference_mode():
            with torch.autocast(self.device_id.split(':')[0]):
                output = self.pipe(
                    prompt=config['pos_prompt'],
                    negative_prompt=config['neg_prompt'],
                    image=pil_img,
                    mask_image=pil_mask,
                    strength=strength,
                    width=new_w,
                    height=new_h,
                    generator=generator,
                    **control_args
                ).images[0]

        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        res_np = cv2.resize(res_np, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
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

            ckpt_path = cfg.get_path('checkpoint', 'checkpoint_file')
            vae_path = cfg.get_path('vae', 'vae_file')
            
            controlnet = None
            if use_controlnet:
                cn_path = cfg.get_path('controlnet', 'controlnet_tile')
                if cn_path and os.path.exists(cn_path):
                    controlnet = ControlNetModel.from_single_file(cn_path, torch_dtype=torch.float16)

            vae = None
            if vae_path and os.path.exists(vae_path):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)

            load_args = {"torch_dtype": torch.float32, "safety_checker": None, "use_safetensors": True}
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