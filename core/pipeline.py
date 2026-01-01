import os
import re
import cv2
import gc
import numpy as np
import torch
from PIL import Image
import traceback

from core.mask_utils import MaskUtils
from core.detector import ObjectDetector
from core.sam_wrapper import SamInference
from core.config import config_instance as cfg
from core.model_manager import ModelManager
from core.face_restorer import FaceRestorer
from core.visualizer import draw_detections, draw_mask_on_image
from core.geometry import align_and_crop, restore_and_paste, is_anatomically_correct
from core.box_sorter import sort_boxes
from core.color_fix import apply_color_fix

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
            
            # [Fix] 로그에 실제 패스 이름 표시 (Unit 1, 2... 대신 패스 2, 3... 표시)
            unit_name = config.get('unit_name', f"Unit {i+1}")
            self.log(f"  > Processing {unit_name} ({config['detector_model']})...")
            
            # [Debug] 현재 패스 설정값 로그 출력
            if cfg.get('system', 'log_level') == 'DEBUG':
                self.log(f"    [Debug] Configuration for {unit_name}:")
                for k, v in config.items():
                    if k in ['pos_prompt', 'neg_prompt']:
                        v_str = str(v).replace('\n', ' ')
                        if len(v_str) > 60: v_str = v_str[:57] + "..."
                        self.log(f"      - {k}: {v_str}")
                    else:
                        self.log(f"      - {k}: {v}")

            try:
                # 모델 로딩 위임
                # 1. 체크포인트/VAE 경로 결정
                # [Fix] Global vs Local 모델 결정 로직 강화
                ckpt_name = None
                if config.get('sep_ckpt') and config.get('sep_ckpt_name') and config['sep_ckpt_name'] != "Use Global":
                    ckpt_name = config['sep_ckpt_name']
                else:
                    ckpt_name = config.get('global_ckpt_name')

                ckpt_path = os.path.join(cfg.get_path('checkpoint'), ckpt_name) if ckpt_name else None
                
                vae_name = None
                if config.get('sep_vae') and config.get('sep_vae_name') and config['sep_vae_name'] != "Use Global":
                    vae_name = config['sep_vae_name']
                else:
                    vae_name = config.get('global_vae_name')
                
                vae_path = os.path.join(cfg.get_path('vae'), vae_name) if vae_name and vae_name != "Automatic" else None

                # 2. ControlNet 경로 결정
                cn_path = None
                if config.get('use_controlnet') and config.get('control_model') != "None":
                    cn_path = os.path.join(cfg.get_path('controlnet'), config['control_model'])

                clip_skip = int(config.get('clip_skip', 1)) if config.get('sep_clip') else 1

                # [Fix] 모델 로드 전 메모리 정리
                gc.collect()
                torch.cuda.empty_cache()

                self.model_manager.load_sd_model(ckpt_path, vae_path, cn_path, clip_skip)
                
                # [Fix] VAE OOM 방지: Tiling 및 Slicing 활성화
                # SDXL 등 고해상도 모델에서 VAE 인코딩/디코딩 시 메모리 부족을 방지하는 핵심 설정
                if hasattr(self.model_manager.pipe.vae, "enable_tiling"):
                    self.model_manager.pipe.vae.enable_tiling()
                elif hasattr(self.model_manager.pipe, "enable_vae_tiling"): # Fallback for older diffusers
                    self.model_manager.pipe.enable_vae_tiling()

                if hasattr(self.model_manager.pipe.vae, "enable_slicing"):
                    self.model_manager.pipe.vae.enable_slicing()
                elif hasattr(self.model_manager.pipe, "enable_vae_slicing"): # Fallback for older diffusers
                    self.model_manager.pipe.enable_vae_slicing()
                
                try:
                    result_img = self._process_pass(result_img, config)
                    if self.preview_callback:
                        self.preview_callback(result_img)
                except Exception as e:
                    self.log(f"  [Error] Failed to process {unit_name}: {e}")
                    traceback.print_exc()
            except Exception as e:
                self.log(f"  [Error] Setup failed for {unit_name}: {e}")
                traceback.print_exc()
            finally:
                self.model_manager.manage_lora([], action="unload")
                # [Fix] 패스 종료 시 메모리 정리
                gc.collect()
                torch.cuda.empty_cache()
                
        return result_img

    def _process_pass(self, image, config):
        h, w = image.shape[:2]
        img_area = h * w
        
        detections = self.detector.detect(image, config['detector_model'], config['conf_thresh'])
        if not detections:
            self.log(f"    [Info] No objects detected (Threshold: {config['conf_thresh']}).")
            return image

        # [New] Split Prompts by [SEP] (ADetailer Syntax)
        sep_pattern = r"\s*\[SEP\]\s*"
        pos_prompts = re.split(sep_pattern, config.get('pos_prompt', ''))
        neg_prompts = re.split(sep_pattern, config.get('neg_prompt', ''))

        # [Modified] Sort Detections based on Config
        boxes = [d['box'] for d in detections]
        scores = [d['conf'] for d in detections]
        sort_method = config.get('sort_method', '신뢰도')
        _, _, sorted_indices = sort_boxes(boxes, scores, sort_method, w, h)
        detections = [detections[i] for i in sorted_indices]

        # [New] Apply Max Detections Limit (정렬 후 상위 N개만 선택)
        max_det = config.get('max_det', 20)
        if max_det > 0 and len(detections) > max_det:
            detections = detections[:max_det]

        # [New] Pre-calculate LoRAs for visualization (프리뷰용 LoRA 정보 미리 계산)
        for i, det in enumerate(detections):
            cur_pos = pos_prompts[i] if i < len(pos_prompts) else pos_prompts[-1]
            _, lora_list = self._parse_and_extract_loras(cur_pos)
            det['lora_names'] = [name for name, _ in lora_list]

        # [Fix] 정렬 및 필터링이 끝난 후 프리뷰를 생성해야 실제 처리 순서와 일치함
        if self.preview_callback:
            vis_img = draw_detections(image, detections)
            self.preview_callback(vis_img)

        if config['use_sam']:
            if self.sam is None:
                sam_file = cfg.get_path('sam', 'sam_file')
                self.sam = SamInference(checkpoint=sam_file, device=self.device)
            self.sam.set_image(image)

        final_img = image.copy()

        for i, det in enumerate(detections):
            box = det['box']
            x1, y1, x2, y2 = box
            
            # Area Filtering
            face_ratio = ((box[2]-x1)*(y2-y1)) / img_area
            if face_ratio < config['min_face_ratio']:
                self.log(f"    Skipping detection #{i+1}: Too small ({face_ratio:.4f} < {config['min_face_ratio']})")
                continue
            if face_ratio > config['max_face_ratio']:
                self.log(f"    Skipping detection #{i+1}: Too large ({face_ratio:.4f} > {config['max_face_ratio']})")
                continue

            # [New] Gender Filter (성별 필터)
            target_gender = config.get('gender_filter', 'All')
            if target_gender != 'All':
                detected_gender = self.detector.analyze_gender(image, box)
                if detected_gender and detected_gender != target_gender:
                    self.log(f"  Skipping detection: Gender mismatch ({detected_gender} != {target_gender})")
                    continue

            # [New] Select Prompt for this object (ADetailer Logic)
            # If there are more detected objects than separate prompts, the last prompt will be used.
            cur_pos = pos_prompts[i] if i < len(pos_prompts) else pos_prompts[-1]
            cur_neg = neg_prompts[i] if i < len(neg_prompts) else neg_prompts[-1]

            # [Debug] 프롬프트 할당 확인 로그 (검증용)
            if cfg.get('system', 'log_level') == 'DEBUG':
                p_idx = i if i < len(pos_prompts) else len(pos_prompts) - 1
                self.log(f"    [Debug] Object #{i+1}: Applied Prompt Segment #{p_idx + 1}")

            # [New] Extract LoRAs for THIS detection (Dynamic Loading)
            # 각 객체마다 다른 LoRA를 적용하기 위해 여기서 파싱 및 로드 수행
            clean_pos, lora_list = self._parse_and_extract_loras(cur_pos)
            # ModelManager가 스마트 캐싱을 수행하므로, 이전과 같은 LoRA면 재로딩하지 않음
            self.model_manager.manage_lora(lora_list, action="load")

            # [New] Check [SKIP]
            if re.match(r"^\s*\[SKIP\]\s*$", cur_pos, re.IGNORECASE):
                self.log(f"  Skipping detection {i+1}: [SKIP] token found.")
                continue

            # [New] Replace [PROMPT] token
            # Standalone 환경에서는 원본 생성 프롬프트가 없으므로 빈 문자열로 대체하여 토큰 오염 방지
            clean_pos = clean_pos.replace("[PROMPT]", "")
            cur_neg = cur_neg.replace("[PROMPT]", "")
            
            # Create config for this specific detection
            det_config = config.copy()
            det_config['pos_prompt'] = clean_pos
            det_config['neg_prompt'] = cur_neg

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
            
            # [New] 실시간 마스크 확인: 인페인팅 직전에 현재 적용될 마스크를 프리뷰에 표시
            if self.preview_callback:
                mask_vis = draw_mask_on_image(final_img, mask, color=(0, 255, 0))
                self.preview_callback(mask_vis)
            
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
            final_img = self._run_inpaint(final_img, mask, det_config, denoise, box, kps)

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
        
        target_res = 512
        # [Fix] SDXL Model Detection for Default Resolution
        if hasattr(self.model_manager.pipe, "tokenizer_2"):
            target_res = 1024
            
        if config.get('inpaint_width', 0) > 0: target_res = config['inpaint_width']
        
        do_rotate = config.get('auto_rotate', False) and kps is not None
        
        # 1. Align & Crop (Image & Mask)
        proc_img, M = align_and_crop(image, box, kps, target_size=target_res, padding=padding_ratio, force_rotate=do_rotate)
        proc_mask, _ = align_and_crop(mask, box, kps, target_size=target_res, padding=padding_ratio, force_rotate=do_rotate, borderMode=cv2.BORDER_CONSTANT)
        
        # [Fix] 합성 시 바운딩 박스 흔적을 없애기 위해 Soft Mask 보존
        paste_mask = proc_mask.copy()

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
        # [Fix] 인퍼런스 직전 메모리 정리
        torch.cuda.empty_cache()
        
        with torch.inference_mode():
            with torch.autocast(self.device.split(':')[0]):
                output = self.model_manager.pipe(**infer_args).images[0]

        # Paste Back (Alpha Blend)
        res_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        
        # [New] Color Fix (색감 보정)
        color_fix_method = config.get('color_fix', 'None')
        if color_fix_method != 'None':
            res_np = apply_color_fix(res_np, proc_img, color_fix_method)
        
        # Restore Face (얼굴 보정)
        if config.get('restore_face', False):
            res_np = self.face_restorer.restore(res_np)

        # [Fix] Use Geometry Module for Inverse Transform & Blending
        final_img = restore_and_paste(image, res_np, M, mask_blur=config['mask_blur'], paste_mask=paste_mask)
        
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