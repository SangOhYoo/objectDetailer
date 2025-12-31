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

# Diffusers Imports
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel
)

# --- Global Resources ---
_global_load_lock = threading.Lock() # 모델 로딩/스위칭 시 충돌 방지 (Thread Safety)
_job_queue = Queue() # 작업 대기열 (Producer-Consumer Pattern)

class ProcessingController(QObject):
    """
    메인 윈도우와 통신하며 GPU Worker들을 관리하고 작업을 분배하는 컨트롤러
    """
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
        # 1. 작업 큐 채우기
        for f in self.file_paths:
            _job_queue.put(f)

        # 2. GPU 감지 및 워커 생성
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log_signal.emit(f"[System] Detected {gpu_count} NVIDIA GPUs.")
        else:
            gpu_count = 1
            self.log_signal.emit("[System] No GPU detected. Using CPU (Very Slow).")

        for i in range(gpu_count):
            # GPU ID 할당 (cuda:0, cuda:1 ...)
            dev_id = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            worker = GpuWorker(dev_id, self.configs, worker_id=i)
            
            # 시그널 연결
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
        # 모든 파일 처리가 완료되었는지 확인
        if self.processed_count >= self.total_files:
            self.finished_signal.emit()


class GpuWorker(QThread):
    """
    개별 GPU에서 독립적으로 돌아가는 워커 스레드.
    Detector -> SAM -> Inpaint 파이프라인을 수행합니다.
    """
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str, np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, device_id, configs, worker_id):
        super().__init__()
        self.device_id = device_id
        self.configs = configs
        self.worker_id = worker_id
        
        # 모델 인스턴스 홀더
        self.pipe = None
        self.detector = ObjectDetector(device=device_id)
        self.sam = None # Lazy Loading

        # 기본 모델 경로 설정
        self.sd_model_path = "runwayml/stable-diffusion-inpainting"
        self.lora_dir = os.path.join("models", "loras")
        os.makedirs(self.lora_dir, exist_ok=True)

    def run(self):
        self.log_signal.emit(f"[Worker-{self.worker_id}] Initialized on {self.device_id}")

        while not _job_queue.empty():
            try:
                # 큐에서 파일 경로 가져오기 (Non-blocking)
                fpath = _job_queue.get_nowait()
            except:
                break

            self.log_signal.emit(f"[Worker-{self.worker_id}] Processing: {os.path.basename(fpath)}")
            try:
                # 1. 이미지 로드 [FIX: 한글 경로 호환성]
                # numpy로 바이너리를 읽은 후 디코딩해야 한글 경로에서 에러가 안 남
                img_stream = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                
                if img is None:
                    self.log_signal.emit(f"[Worker-{self.worker_id}] Error: Failed to load image (Check path).")
                    continue
                
                # 2. 이미지 처리 파이프라인 실행
                result_img = self.process_image(img)
                
                # 3. 결과 전송
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
        """
        설정된 Unit(탭)들을 순차적으로 실행
        """
        result_img = image.copy()
        
        for i, config in enumerate(self.configs):
            if not config['enabled']:
                continue
            
            unit_name = f"Unit {i+1}"
            self.log_signal.emit(f"  > [Worker-{self.worker_id}] Running {unit_name} ({config['model']})...")
            
            # 1. 파이프라인 로드 (BMAB: ControlNet 사용 여부 확인)
            self.load_sd_model(use_controlnet=config['use_controlnet'])
            
            # 2. LoRA 로드 (BMAB: Pass별 LoRA 주입)
            self.manage_lora(config, action="load")

            try:
                # 3. 단일 패스 실행
                result_img = self.process_pass(result_img, config)
            finally:
                # 4. LoRA 언로드 (다음 패스를 위해 정리)
                self.manage_lora(config, action="unload")
            
        return result_img

    def process_pass(self, image, config):
        """
        [Core Logic] Detect -> Mask -> Inpaint
        """
        h, w = image.shape[:2]
        img_area = h * w
        
        # A. 탐지 (Detector)
        # Returns list of dicts: {'box': [x1,y1,x2,y2], 'mask': np.array, ...}
        detections = self.detector.detect(image, config['model'], config['conf'])
        
        if not detections:
            return image

        # 면적순 정렬 (큰 객체부터 처리)
        detections.sort(key=lambda d: (d['box'][2]-d['box'][0]) * (d['box'][3]-d['box'][1]), reverse=True)

        # SAM 초기화 (필요한 경우)
        if config['use_sam']:
            if self.sam is None:
                with _global_load_lock:
                    self.sam = SamInference(device=self.device_id)
            # 이미지 임베딩 (한 번만 수행)
            self.sam.set_image(image)

        final_img = image.copy()

        # B. 객체별 처리 루프
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            
            # --- [FIX: Area Filtering] ---
            # 너무 작거나 큰 객체는 무시 (ADetailer 핵심 기능)
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / img_area
            
            if ratio < config['min_area'] or ratio > config['max_area']:
                # self.log_signal.emit(f"    Skipped object ratio {ratio:.3f} (Out of range)")
                continue
            # -----------------------------

            yolo_mask = det['mask'] # YOLO Seg 모델일 경우 존재

            # --- 1. 마스크 생성 ---
            if config['use_sam'] and self.sam:
                # [Option 1] SAM 정밀 누끼
                mask = self.sam.predict_mask_from_box(box)
            elif yolo_mask is not None:
                # [Option 2] YOLO Segmentation 결과 사용
                mask = yolo_mask
            else:
                # [Option 3] 단순 박스 (Fallback)
                mask = MaskUtils.box_to_mask(box, (h, w), padding=0)

            # --- 2. 마스크 후가공 (ADetailer Logic) ---
            mask = MaskUtils.refine_mask(
                mask, 
                dilation=config['dilation'], 
                blur=config['blur']
            )

            # --- 3. Dynamic Denoising (BMAB Logic) ---
            final_denoise = self._calc_dynamic_denoise(box, (h, w), config['denoise'])

            # --- 4. 인페인팅 실행 ---
            final_img = self.run_inpaint(final_img, mask, config, final_denoise)

        return final_img

    def run_inpaint(self, image, mask, config, strength):
        """
        Crop -> Resize -> Inpaint -> Paste
        """
        # 1. Crop (Context Padding 포함)
        padding = config['padding']
        crop_img, (x1, y1, x2, y2) = MaskUtils.crop_image_by_mask(image, mask, context_padding=padding)
        crop_mask = mask[y1:y2, x1:x2]
        
        if crop_img.size == 0: return image

        # 2. Convert to PIL & RGB
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(crop_mask)

        # 3. ControlNet Preprocessing (Canny)
        control_args = {}
        if config['use_controlnet']:
            # Canny Edge 추출
            canny = cv2.Canny(crop_img, 100, 200)
            # Diffusers ControlNet expects 3-channel RGB image
            canny = np.stack([canny] * 3, axis=-1)
            pil_control = Image.fromarray(canny)
            
            control_args["control_image"] = pil_control
            control_args["controlnet_conditioning_scale"] = config['cn_weight']

        # 4. Inference
        with torch.inference_mode():
            with torch.autocast(self.device_id.split(':')[0]): # "cuda"
                output = self.pipe(
                    prompt=config['pos_prompt'],
                    negative_prompt=config['neg_prompt'],
                    image=pil_img,
                    mask_image=pil_mask,
                    strength=strength,
                    # SD 1.5는 8배수 해상도 필요 (Resize safety)
                    width=pil_img.width - (pil_img.width % 8),
                    height=pil_img.height - (pil_img.height % 8),
                    **control_args
                ).images[0]

        # 5. Paste Back
        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        # 추론 시 8배수로 잘렸을 수 있으므로 다시 원본 크롭 크기로 리사이즈
        res_np = cv2.resize(res_np, (x2 - x1, y2 - y1))
        
        # 단순 붙여넣기 (Alpha Blending 권장되나 성능상 Copy 사용)
        image[y1:y2, x1:x2] = res_np
        
        return image

    def _calc_dynamic_denoise(self, box, img_shape, base_strength):
        """
        객체가 작을수록(멀수록) 디테일 재창조를 위해 강도 높임
        """
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        img_area = img_shape[0] * img_shape[1]
        ratio = box_area / img_area
        
        adjustment = 0.0
        if ratio < 0.05: adjustment = 0.15 # 5% 미만: 매우 작음
        elif ratio < 0.10: adjustment = 0.10 # 10% 미만: 작음
        elif ratio < 0.20: adjustment = 0.05 # 20% 미만: 중간
        
        return max(0.1, min(base_strength + adjustment, 0.8))

    def load_sd_model(self, use_controlnet=False):
        """
        Global Lock을 사용하여 안전하게 모델 로드/교체
        """
        # 이미 적절한 파이프라인이 로드되어 있으면 패스
        is_cn_loaded = isinstance(self.pipe, StableDiffusionControlNetInpaintPipeline)
        if self.pipe is not None and is_cn_loaded == use_controlnet:
            return

        with _global_load_lock:
            # 메모리 정리
            if self.pipe:
                del self.pipe
            torch.cuda.empty_cache()

            # 1. ControlNet 로드
            controlnet = None
            if use_controlnet:
                # [주의] 실제 사용 시엔 로컬 경로 확인 로직 추가 권장
                cn_repo = "lllyasviel/control_v11p_sd15_canny"
                controlnet = ControlNetModel.from_pretrained(cn_repo, torch_dtype=torch.float16)

            # 2. 파이프라인 설정
            load_args = {"torch_dtype": torch.float32, "safety_checker": None}
            if controlnet:
                PipelineClass = StableDiffusionControlNetInpaintPipeline
                load_args["controlnet"] = controlnet
            else:
                PipelineClass = StableDiffusionInpaintPipeline

            # 3. 모델 로드 (로컬 경로 우선, 없으면 HF 다운로드)
            # 여기서는 편의상 from_pretrained 사용 (실제론 로컬 경로 사용 권장)
            self.pipe = PipelineClass.from_pretrained(self.sd_model_path, **load_args)

            # 4. GPU 이동 및 FP16 변환
            self.pipe.to(self.device_id)
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.text_encoder.to(dtype=torch.float16)
            if controlnet:
                self.pipe.controlnet.to(dtype=torch.float16)
            self.pipe.vae.to(dtype=torch.float32) # VAE는 FP32 유지 (NaN 방지)

    def manage_lora(self, config, action="load"):
        """
        [FIX: LoRA 안전 관리] 파일 부재 시 멈추지 않고 스킵
        """
        lora_name = config.get('lora_model', 'None')
        if lora_name == "None":
            return

        try:
            if action == "load":
                # LoRA 파일 경로 확인
                lora_path = os.path.join(self.lora_dir, lora_name)
                
                if not os.path.exists(lora_path):
                    self.log_signal.emit(f"    [Warning] LoRA not found: {lora_name}. Skipping.")
                    return

                # adapter_name을 지정하여 나중에 특정해서 지울 수 있게 함
                self.pipe.load_lora_weights(lora_path, adapter_name="default")
                self.pipe.fuse_lora(lora_scale=config['lora_scale'])
                self.log_signal.emit(f"    [LoRA] Injected: {lora_name}")

            elif action == "unload":
                # LoRA 제거 (메모리 및 다음 Pass 영향 방지)
                self.pipe.unfuse_lora()
                self.pipe.unload_lora_weights()
                
        except Exception as e:
            self.log_signal.emit(f"    [LoRA Error] Failed to {action}: {str(e)}")