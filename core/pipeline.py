import os
import re
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
from core.visualizer import draw_detections
from core.geometry import align_and_crop, restore_and_paste, is_anatomically_correct

class ImageProcessor:
    def __init__(self, device, log_callback=None, preview_callback=None):
        self.device = device
        self.log_callback = log_callback
        self.preview_callback = preview_callback
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
            
            self.log(f"  > Processing Unit {i+1} ({config['detector_model']})...")
            
            # 모델 로딩 위임
            # 1. 체크포인트/VAE 경로 결정
            ckpt_path = None
            if config.get('sep_ckpt') and config.get('sep_ckpt_name') and config['sep_ckpt_name'] != "Use Global":
                ckpt_path = os.path.join(cfg.get_path('checkpoint'), config['sep_ckpt_name'])
            
            vae_path = None
            if config.get('sep_vae') and config.get('sep_vae_name') and config['sep_vae_name'] != "Use Global":
                vae_path = os.path.join(cfg.get_path('vae'), config['sep_vae_name'])

            # 2. ControlNet 경로 결정
            cn_path = None
            if config.get('use_controlnet') and config.get('control_model') != "None":
                cn_path = os.path.join(cfg.get_path('controlnet'), config['control_model'])

            clip_skip = int(config.get('clip_skip', 1)) if config.get('sep_clip') else 1

            self.model_manager.load_sd_model(ckpt_path, vae_path, cn_path, clip_skip)
            
            # [New] Prompt 기반 LoRA 파싱 및 로드
            raw_prompt = config.get('pos_prompt', '')
            clean_prompt, lora_list = self._parse_and_extract_loras(raw_prompt)
            
            self.model_manager.manage_lora(lora_list, action="load")
            
            # 이번 패스에서만 사용할 Clean Prompt 적용 (Config 복사본 사용)
            pass_config = config.copy()
            pass_config['pos_prompt'] = clean_prompt

            try:
                result_img = self._process_pass(result_img, pass_config)
                if self.preview_callback:
                    self.preview_callback(result_img)
            finally:
                self.model_manager.manage_lora(lora_list, action="unload")
                
        return result_img

    def _process_pass(self, image, config):
        h, w = image.shape[:2]
        img_area = h * w
        
        detections = self.detector.detect(image, config['detector_model'], config['conf_thresh'])
        if not detections: return image

        # [New] 탐지된 영역(박스+마스크+점수) 시각화 및 프리뷰 전송
        if self.preview_callback:
            vis_img = draw_detections(image, detections)
            self.preview_callback(vis_img)

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
            if (box[2]-x1)*(y2-y1)/img_area < config['min_face_ratio'] or (box[2]-x1)*(y2-y1)/img_area > config['max_face_ratio']:
                continue

            # [New] Gender Filter (성별 필터)
            target_gender = config.get('gender_filter', 'All')
            if target_gender != 'All':
                detected_gender = self.detector.analyze_gender(image, box)
                if detected_gender and detected_gender != target_gender:
                    self.log(f"  Skipping detection: Gender mismatch ({detected_gender} != {target_gender})")
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
            mask = MaskUtils.refine_mask(mask, dilation=config['mask_dilation'], blur=config['mask_blur'])
            # if config.get('merge_mode') == "Merge and Invert":
            #     mask = cv2.bitwise_not(mask)
            
            # Dynamic Denoise
            denoise = self._calc_dynamic_denoise(box, (h, w), config['denoising_strength'])

            # [New] Landmarks for Rotation/Anatomy
            kps = None
            if config.get('auto_rotate') or config.get('anatomy_check'):
                kps = self.detector.get_face_landmarks(image, box)

            if config.get('anatomy_check') and kps:
                if not is_anatomically_correct(kps):
                    self.log(f"  Skipping detection: Anatomically incorrect.")
                    continue
            
            # Inpaint
            final_img = self._run_inpaint(final_img, mask, config, denoise, box, kps)

            # [New] 얼굴 하나 처리될 때마다 프리뷰 갱신 (실시간 피드백)
            if self.preview_callback:
                self.preview_callback(final_img)

        return final_img

    def _run_inpaint(self, image, mask, config, strength, box, kps):
        # [Fix] Use Geometry Module for Alignment & Rotation
        padding_px = config['crop_padding']
        
        # Convert pixel padding to ratio for align_and_crop
        w_box, h_box = box[2] - box[0], box[3] - box[1]
        padding_ratio = (padding_px * 2) / max(w_box, h_box) if max(w_box, h_box) > 0 else 0.25
        
        target_res = 512 # Default processing resolution
        if config.get('inpaint_width', 0) > 0: target_res = config['inpaint_width']
        
        do_rotate = config.get('auto_rotate', False) and kps is not None
        
        # 1. Align & Crop (Image & Mask)
        proc_img, M = align_and_crop(image, box, kps, target_size=target_res, padding=padding_ratio, force_rotate=do_rotate)
        proc_mask, _ = align_and_crop(mask, box, kps, target_size=target_res, padding=padding_ratio, force_rotate=do_rotate, borderMode=cv2.BORDER_CONSTANT)
        
        # Binarize mask after warping
        _, proc_mask = cv2.threshold(proc_mask, 127, 255, cv2.THRESH_BINARY)
        
        new_h, new_w = proc_img.shape[:2]

        pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(proc_mask)

        # ControlNet
        control_args = {}
        # [Fix] 파이프라인에 ControlNet이 로드되어 있다면, 설정(use_controlnet)과 무관하게 이미지를 공급해야 함
        if hasattr(self.model_manager.pipe, "controlnet"):
            if not config['use_controlnet']:
                # ControlNet이 로드되었으나 사용 안 함 -> 가중치 0으로 설정하여 영향력 제거
                control_args["control_image"] = pil_img
                control_args["controlnet_conditioning_scale"] = 0.0
            else:
                cn_model = config.get('control_model', '').lower()
            
                if 'tile' in cn_model:
                    # Tile 모델은 원본 이미지를 그대로 사용 (혹은 블러)
                    control_args["control_image"] = pil_img
                elif 'openpose' in cn_model:
                    # OpenPose 지원
                    try:
                        from controlnet_aux import OpenposeDetector
                        # 매번 로드하면 느리지만, 현재 구조상 여기서 처리 (추후 캐싱 필요)
                        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                        openpose.to(self.device)
                        pose_img = openpose(pil_img)
                        control_args["control_image"] = pose_img
                    except ImportError:
                        print("[Pipeline] Warning: controlnet_aux not found. OpenPose disabled.")
                        control_args["control_image"] = pil_img # Fallback
                else:
                    # 기본값: Canny (OpenPose 등은 별도 전처리기 필요하나 여기선 Canny로 fallback)
                    canny = cv2.Canny(proc_img, 100, 200)
                    canny = np.stack([canny] * 3, axis=-1)
                    control_args["control_image"] = Image.fromarray(canny)
            
                control_args["controlnet_conditioning_scale"] = float(config['control_weight'])
                control_args["control_guidance_start"] = float(config.get('guidance_start', 0.0))
                control_args["control_guidance_end"] = float(config.get('guidance_end', 1.0))

        # [Fix] Long Prompt Support (Token Chunking)
        # SDXL은 2개의 텍스트 인코더를 사용하므로 기존 로직(SD1.5용)과 호환되지 않음.
        # SDXL일 경우(tokenizer_2 존재) 수동 임베딩 생성을 건너뛰고 파이프라인에 맡김.
        if hasattr(self.model_manager.pipe, "tokenizer_2"):
            prompt_embeds, neg_prompt_embeds = None, None
        else:
            prompt_embeds, neg_prompt_embeds = self._get_long_prompt_embeds(
                self.model_manager.pipe, config['pos_prompt'], config['neg_prompt']
            )

        # Apply Scheduler & Seed
        self.model_manager.apply_scheduler(config.get('sampler_name', 'Euler a'))
        seed = config.get('seed', -1)
        generator = torch.Generator(self.device)
        if seed != -1: generator.manual_seed(seed)

        infer_args = {
            "image": pil_img,
            "mask_image": pil_mask,
            "strength": strength,
            "width": new_w, "height": new_h,
            "generator": generator,
            **control_args
        }
        
        if prompt_embeds is not None:
            infer_args["prompt_embeds"] = prompt_embeds
            infer_args["negative_prompt_embeds"] = neg_prompt_embeds
        else:
            infer_args["prompt"] = config['pos_prompt']
            infer_args["negative_prompt"] = config['neg_prompt']

        # Inference
        with torch.inference_mode():
            with torch.autocast(self.device.split(':')[0]):
                output = self.model_manager.pipe(**infer_args).images[0]

        # Paste Back (Alpha Blend)
        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        
        # Restore Face (얼굴 보정)
        if config.get('restore_face', False):
            res_np = self.face_restorer.restore(res_np)

        # [Fix] Use Geometry Module for Inverse Transform & Blending
        final_img = restore_and_paste(image, res_np, M, mask_blur=config['mask_blur'])
        
        return final_img

    def _calc_dynamic_denoise(self, box, img_shape, base):
        x1, y1, x2, y2 = box
        ratio = ((x2 - x1) * (y2 - y1)) / (img_shape[0] * img_shape[1])
        adj = 0.15 if ratio < 0.05 else (0.10 if ratio < 0.10 else (0.05 if ratio < 0.20 else 0.0))
        return max(0.1, min(base + adj, 0.8))

    def _get_long_prompt_embeds(self, pipe, prompt, negative_prompt):
        """77토큰 제한을 우회하기 위한 임베딩 청킹(Chunking) 처리"""
        if not prompt: prompt = ""
        if not negative_prompt: negative_prompt = ""
        
        # 파이프라인에 토크나이저/인코더가 없는 경우
        if not hasattr(pipe, "tokenizer") or not hasattr(pipe, "text_encoder"):
            return None, None
            
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        
        if not tokenizer or not text_encoder:
            return None, None

        # 1. Tokenize
        pos_tokens = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        neg_tokens = tokenizer(negative_prompt, truncation=False, add_special_tokens=False).input_ids

        # 2. Chunking
        max_len = tokenizer.model_max_length - 2
        
        def chunk_tokens(tokens):
            if len(tokens) == 0: return [[]]
            return [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]

        pos_chunks = chunk_tokens(pos_tokens)
        neg_chunks = chunk_tokens(neg_tokens)

        # 3. Pad chunks to match max length (Reforge style)
        total_chunks = max(len(pos_chunks), len(neg_chunks))
        
        while len(pos_chunks) < total_chunks:
            pos_chunks.append([])
        while len(neg_chunks) < total_chunks:
            neg_chunks.append([])

        # 4. Encode
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def encode(chunks):
            embeds = []
            for chunk in chunks:
                input_ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
                pad_len = tokenizer.model_max_length - len(input_ids)
                if pad_len > 0:
                    input_ids += [pad_token] * pad_len
                
                # [Fix] CPU Offload 호환성: 모델이 CPU에 있으면 입력도 CPU로 생성
                input_tensor = torch.tensor([input_ids], device=text_encoder.device)
                embeds.append(text_encoder(input_tensor)[0])
            return torch.cat(embeds, dim=1).to(self.device)

        return encode(pos_chunks), encode(neg_chunks)

    def _parse_and_extract_loras(self, prompt):
        """
        프롬프트에서 <lora:filename:multiplier> 태그를 추출하고 제거합니다.
        반환값: (clean_prompt, [(name, scale), ...])
        """
        # 정규식: <lora:이름> 또는 <lora:이름:강도>
        pattern = r"<lora:([^:>]+)(?::([\d.]+))?>"
        loras = []
        
        def replace_func(match):
            name = match.group(1)
            scale = float(match.group(2)) if match.group(2) else 1.0
            loras.append((name, scale))
            return "" # 태그 제거
            
        clean_prompt = re.sub(pattern, replace_func, prompt)
        return clean_prompt, loras