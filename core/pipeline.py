import cv2
import numpy as np
import torch
from PIL import Image

from core.mask_utils import MaskUtils
from core.detector import ObjectDetector
from core.sam_wrapper import SamInference
from core.config import config_instance as cfg
from core.model_manager import ModelManager
from core.face_restorer import FaceRestorer

class ImageProcessor:
    def __init__(self, device, log_callback=None):
        self.device = device
        self.log_callback = log_callback
        self.model_manager = ModelManager(device)
        
        # Detectors
        model_dir = cfg.get_path('sam')
        self.detector = ObjectDetector(device=device, model_dir=model_dir)
        self.sam = None
        self.face_restorer = FaceRestorer(device)

    def log(self, msg):
        if self.log_callback: self.log_callback(msg)

    def process(self, image, configs):
        result_img = image.copy()
        
        for i, config in enumerate(configs):
            if not config['enabled']: continue
            
            self.log(f"  > Processing Unit {i+1} ({config['model']})...")
            
            # 모델 로딩 위임
            # 1. 체크포인트/VAE 경로 결정
            ckpt_path = None
            if config.get('sep_ckpt') and config.get('sep_ckpt_name'):
                ckpt_path = os.path.join(cfg.get_path('checkpoint'), config['sep_ckpt_name'])
            
            vae_path = None
            if config.get('sep_vae') and config.get('sep_vae_name'):
                vae_path = os.path.join(cfg.get_path('vae'), config['sep_vae_name'])

            # 2. ControlNet 경로 결정
            cn_path = None
            if config.get('use_controlnet') and config.get('cn_model') != "None":
                cn_path = os.path.join(cfg.get_path('controlnet'), config['cn_model'])

            clip_skip = int(config.get('clip_skip', 1)) if config.get('sep_clip') else 1

            self.model_manager.load_sd_model(ckpt_path, vae_path, cn_path, clip_skip)
            self.model_manager.manage_lora(config, action="load")

            try:
                result_img = self._process_pass(result_img, config)
            finally:
                self.model_manager.manage_lora(config, action="unload")
                
        return result_img

    def _process_pass(self, image, config):
        h, w = image.shape[:2]
        img_area = h * w
        
        detections = self.detector.detect(image, config['model'], config['conf'])
        if not detections: return image

        detections.sort(key=lambda d: (d['box'][2]-d['box'][0]) * (d['box'][3]-d['box'][1]), reverse=True)

        if config['use_sam']:
            if self.sam is None:
                sam_file = cfg.get_path('sam', 'sam_file')
                self.sam = SamInference(checkpoint=sam_file, device=self.device)
            self.sam.set_image(image)

        final_img = image.copy()

        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            
            # Area Filtering
            if (box[2]-x1)*(y2-y1)/img_area < config['min_area'] or (box[2]-x1)*(y2-y1)/img_area > config['max_area']:
                continue

            # Masking
            if config['use_sam'] and self.sam:
                mask = self.sam.predict_mask_from_box(box)
            elif det['mask'] is not None:
                mask = det['mask']
            else:
                mask = MaskUtils.box_to_mask(box, (h, w), padding=0)

            # Mask Refine
            mask = MaskUtils.shift_mask(mask, config.get('x_offset', 0), config.get('y_offset', 0))
            mask = MaskUtils.refine_mask(mask, dilation=config['dilation'], blur=config['blur'])
            if config.get('merge_mode') == "Merge and Invert":
                mask = cv2.bitwise_not(mask)
            
            # Dynamic Denoise
            denoise = self._calc_dynamic_denoise(box, (h, w), config['denoise'])
            
            # Inpaint
            final_img = self._run_inpaint(final_img, mask, config, denoise)

        return final_img

    def _run_inpaint(self, image, mask, config, strength):
        padding = config['padding']
        crop_img, (x1, y1, x2, y2) = MaskUtils.crop_image_by_mask(image, mask, context_padding=padding)
        crop_mask = mask[y1:y2, x1:x2]
        
        if crop_img.size == 0: return image

        # Upscale Logic (High-Res Fix)
        h_orig, w_orig = crop_img.shape[:2]
        target_res = 512
        
        if max(h_orig, w_orig) < target_res:
            scale = target_res / max(h_orig, w_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
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

        # ControlNet
        control_args = {}
        if config['use_controlnet']:
            cn_model = config.get('cn_model', '').lower()
            
            if 'tile' in cn_model:
                # Tile 모델은 원본 이미지를 그대로 사용 (혹은 블러)
                control_args["control_image"] = pil_img
            else:
                # 기본값: Canny (OpenPose 등은 별도 전처리기 필요하나 여기선 Canny로 fallback)
                canny = cv2.Canny(proc_img, 100, 200)
                canny = np.stack([canny] * 3, axis=-1)
                control_args["control_image"] = Image.fromarray(canny)
            
            control_args["controlnet_conditioning_scale"] = float(config['cn_weight'])

        # Apply Scheduler & Seed
        self.model_manager.apply_scheduler(config.get('sampler', 'Euler a'))
        seed = config.get('seed', -1)
        generator = torch.Generator(self.device)
        if seed != -1: generator.manual_seed(seed)

        # Inference
        with torch.inference_mode():
            with torch.autocast(self.device.split(':')[0]):
                output = self.model_manager.pipe(
                    prompt=config['pos_prompt'],
                    negative_prompt=config['neg_prompt'],
                    image=pil_img,
                    mask_image=pil_mask,
                    strength=strength,
                    width=new_w, height=new_h,
                    generator=generator,
                    **control_args
                ).images[0]

        # Paste Back (Alpha Blend)
        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        res_np = cv2.resize(res_np, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
        
        alpha = crop_mask.astype(float) / 255.0
        
        # Restore Face (얼굴 보정)
        if config.get('restore_face'):
            res_np = self.face_restorer.restore(res_np)

        alpha = cv2.merge([alpha, alpha, alpha])
        blended = res_np.astype(float) * alpha + crop_img.astype(float) * (1.0 - alpha)
        
        image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return image

    def _calc_dynamic_denoise(self, box, img_shape, base):
        x1, y1, x2, y2 = box
        ratio = ((x2 - x1) * (y2 - y1)) / (img_shape[0] * img_shape[1])
        adj = 0.15 if ratio < 0.05 else (0.10 if ratio < 0.10 else (0.05 if ratio < 0.20 else 0.0))
        return max(0.1, min(base + adj, 0.8))