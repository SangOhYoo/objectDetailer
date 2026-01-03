import os
import re
import cv2
import math
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

        if not detections:
            self.log(f"    [Info] No objects detected (Threshold: {config['conf_thresh']}).")
            return image

    def _resize_by_person(self, image, config):
        """
        [BMAB Feature] Resize by Person
        Expand canvas if person height ratio exceeds target ratio.
        """
        h, w = image.shape[:2]
        target_ratio = config.get('resize_ratio', 0.6)
        
        # 1. Detect Person (Force 'person' class)
        # Check if the main model is likely restricted to faces/hands. 
        # If so, switch to a generic YOLO model that knows 'person'.
        resize_model = config['detector_model']
        if 'face' in resize_model.lower() or 'hand' in resize_model.lower():
             resize_model = 'yolov8n.pt' # Standard YOLO model usually available or auto-downloaded
             
        # We need a separate detection for this because the main loop might be detecting 'face'
        person_detections = self.detector.detect(image, resize_model, 0.35, classes=['person']) # Use generic conf 0.35
        
        if not person_detections:
            return image
            
        # 2. Find Max Person Height
        max_h = 0
        for det in person_detections:
            x1, y1, x2, y2 = det['box']
            ph = y2 - y1
            if ph > max_h:
                max_h = ph
                
        # 3. Check Ratio
        current_ratio = max_h / h
        if current_ratio <= target_ratio:
            return image # No need to resize
            
        self.log(f"    [Resize] Person ratio {current_ratio:.2f} > {target_ratio:.2f}. Expanding canvas.")
        
        # 4. Calculate New Height
        # max_h / new_h = target_ratio  =>  new_h = max_h / target_ratio
        new_h = int(max_h / target_ratio)
        pad_total = new_h - h
        
        if pad_total <= 0: return image

        # 5. Determine Padding (Top/Bottom)
        align = config.get('resize_align', 'Center')
        top, bottom = 0, 0
        if align == 'Bottom': # Person at bottom, add space at Top
            top = pad_total
        elif align == 'Top': # Person at top, add space at Bottom
            bottom = pad_total
        else: # Center
            top = pad_total // 2
            bottom = pad_total - top
            
        # 6. Apply Padding (Reflection as base)
        # Using BORDER_REFLECT_101 for better continuity than zero
        padded_img = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_REFLECT_101)
        
        # 7. Inpaint Padded Area (Outpainting)
        # Create mask for the padded region
        pad_mask = np.zeros((new_h, w), dtype=np.uint8)
        # White on padded areas
        if top > 0: pad_mask[:top, :] = 255
        if bottom > 0: pad_mask[h+top:, :] = 255
        
        # Run Inpaint
        # Use high denoising to halluncinate new background
        # We need a specific prompt. Since we don't have a separate input for background prompt,
        # we'll use the generic negative but keep positive simple or use 'background'.
        # Actually, using the main prompt is usually safe for background extensions if it describes the scene.
        # But 'person' prompts might bleed. 
        # Let's use a generic 'high quality nature background' or just rely on the model seeing the context.
        # We reuse _run_inpaint but we need to mock a 'box' that covers the whole image?
        # Waait, _run_inpaint expects a crop box. We can pass the whole image box.
        
        # Hack: run_inpaint expects a box to Crop. If we pass the whole image box, it processes the whole image.
        full_box = [0, 0, w, new_h]
        
        # Temp config override for outpainting
        outpaint_config = config.copy()
        outpaint_config['crop_padding'] = 0
        outpaint_config['inpaint_width'] = 1024 # Force higher res for full image?
        outpaint_config['inpaint_height'] = 1024 
        outpaint_config['use_controlnet'] = False # Disable CN for background extension to avoid conflict
        
        # Use generic prompt to avoid person ghosts?
        # "background, scenery, detailed"
        # Ideally user should provide this, but for now automated.
        # outpaint_config['pos_prompt'] = "scenery, background, detailed" 
        
        # Call inpaint
        # We pass the WHOLE image and the MASK.
        # Strength 1.0 means generate fully new content in masked area.
        result_padded = self._run_inpaint(padded_img, pad_mask, outpaint_config, 1.0, full_box, None)
        
        return result_padded

    def _process_pass(self, image, config):
        # [New] Resize by Person (Canvas Expansion)
        # Check BEFORE detection
        if config.get('resize_enable', False):
             try:
                image = self._resize_by_person(image, config)
             except Exception as e:
                self.log(f"    [Error] Resize by person failed: {e}")
                traceback.print_exc()

        h, w = image.shape[:2]
        img_area = h * w
        
        detections = self.detector.detect(image, config['detector_model'], config['conf_thresh'], classes=config.get('yolo_classes'))
        
        image_copy = image.copy()
        for i, det in enumerate(detections):
             box = det['box']
             score = det.get('conf', 0.0)
             
             # ... (Mask Generation / Prediction Logic) ...
             
             # Pass score to draw_mask_on_image
             # mask_vis = draw_mask_on_image(final_img, mask, color=(0, 255, 0), box=box, text=f"{score:.2f}")

        # [Disabled] Face Recovery from Pose
        # User requested to remove "artificial" detections.
        """
        if config.get('use_pose_rotation', False):
             try:
                 # ... (Pose Logic Removed/Commented) ...
             except Exception as e:
                 self.log(f"    [Warning] Pose Face Recovery failed: {e}")
        """

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

        # [Mask Merge Implementation]
        merge_mode = config.get('mask_merge_mode', 'None')
        if isinstance(merge_mode, bool): # Handle legacy boolean
            merge_mode = "Merge" if merge_mode else "None"
            
        merged_mask = None
        merge_candidates = [] # Stores (box, prompt, neg_prompt) for merged pass context

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

            # [New] Pose-based Rotation (Lying Body Support)
            # Override KPS with synthesized landmarks from Pose if enabled
            pose_kps_override = None
            if config.get('use_pose_rotation', False):
                 try:
                     # Detect poses if not already done? 
                     # Doing it once per image is better, but here we are in loop.
                     # Cache poses?
                     if not hasattr(self, '_cached_poses') or self._cached_poses_img_id != id(image):
                         self._cached_poses = self.detector.detect_pose(image)
                         self._cached_poses_img_id = id(image)
                     
                     poses = self._cached_poses
                     
                     # Find matching pose
                     cx, cy = (x1+x2)/2, (y1+y2)/2
                     best_pose = None
                     for p in poses:
                         px1, py1, px2, py2 = p['box']
                         # Simple containment check: Face center inside Pose box
                         if px1 < cx < px2 and py1 < cy < py2:
                             best_pose = p
                             break
                     
                     if best_pose:
                         kp = best_pose['keypoints'] # [17, 3] => x, y, conf
                         
                         # User Request: "Vector Directionality Check (Shoulder->Head vs Hip->Legs)"
                         # Vector: Hip Center -> Shoulder Center = "Up" (Head direction)
                         
                         # Indices: 5:LSh, 6:RSh, 11:LHip, 12:RHip (Note: COCO/YOLO index might differ)
                         # YOLO Pose Keypoints (COCO Format):
                         # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 
                         # 5:LSh, 6:RSh, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist
                         # 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
                         
                         l_sh, r_sh = kp[5], kp[6]
                         l_hip, r_hip = kp[11], kp[12]
                         
                         # Check confidence
                         sh_conf = (l_sh[2] > 0.5 and r_sh[2] > 0.5)
                         hip_conf = (l_hip[2] > 0.5 and r_hip[2] > 0.5)
                         
                         body_angle_rad = None
                         
                         # Method 1: Torso Vector (Best for "Up" direction)
                         if sh_conf and hip_conf:
                             sh_center = (l_sh[:2] + r_sh[:2]) / 2
                             hip_center = (l_hip[:2] + r_hip[:2]) / 2
                             
                             # Vector pointing UP (towards Head)
                             up_vec = sh_center - hip_center
                             
                             # If magnitude is too small, ignore
                             if np.linalg.norm(up_vec) > 10:
                                 # Angle of Up Vector
                                 # We want to rotate this vector so it points Up (0, -1)
                                 # Standard atan2 returns angle from X axis (1, 0)
                                 # Rotated Image Angle = - (VectorAngle - (-90 deg))
                                 # Wait, align_and_crop uses 'get_rotation_angle' which calculates "Horizontal Eye Line".
                                 # Face Horizontal is perpendicular to Body Up.
                                 # If Up is (0, -1) [-90 deg], Face Horizontal is (1, 0) [0 deg].
                                 # Relation: FaceAngle = UpAngle + 90 deg.
                                 up_angle = np.arctan2(up_vec[1], up_vec[0])
                                 body_angle_rad = up_angle + (np.pi / 2)
                                 self.log(f"    [Pose] Torso Vector used. (Up Angle: {math.degrees(up_angle):.1f})")
                         
                         # Method 2: Shoulder Line (fallback if hips missing)
                         # L_Sh -> R_Sh vector is roughly "Face Horizontal".
                         elif sh_conf:
                             vec = r_sh[:2] - l_sh[:2]
                             body_angle_rad = np.arctan2(vec[1], vec[0])
                             self.log(f"    [Pose] Shoulder Line used.")
                         
                         if body_angle_rad is not None:
                             # Synthesize KPS for 'align_and_crop'
                             angle_rad = body_angle_rad
                             
                             # Face Center
                             fc = np.array([cx, cy])
                             # Create a fictitious eye distance
                             dist = max(x2-x1, y2-y1) * 0.4 
                             
                             c, s = np.cos(angle_rad), np.sin(angle_rad)
                             R = np.array([[c, -s], [s, c]])
                             
                             # Synthetic Left/Right Eye (aligned with angle)
                             # Standard: Left(-,0), Right(+,0)
                             l_off = np.dot(R, np.array([-dist/2, 0]))
                             r_off = np.dot(R, np.array([dist/2, 0]))
                             
                             # Nose (used for cross product check in get_rotation_angle)
                             # Must be 'below' eyes relative to angle
                             # Standard: Nose(0, +)
                             n_off = np.dot(R, np.array([0, dist * 0.3]))
                             
                             syn_l = fc + l_off
                             syn_r = fc + r_off
                             syn_n = fc + n_off
                             
                             # Mouth (below nose)
                             m_l_off = np.dot(R, np.array([-dist/4, dist*0.8]))
                             m_r_off = np.dot(R, np.array([dist/4, dist*0.8]))
                             syn_ml = fc + m_l_off
                             syn_mr = fc + m_r_off
                             
                             pose_kps_override = np.array([syn_l, syn_r, syn_n, syn_ml, syn_mr])
                             self.log(f"    [Pose] Overriding rotation: {math.degrees(angle_rad):.1f} deg (Vector-based)")

                 except Exception as e:
                     self.log(f"    [Warning] Pose rotation failed: {e}")

            # Prompt & LoRA Logic
            cur_pos = pos_prompts[i] if i < len(pos_prompts) else pos_prompts[-1]
            cur_neg = neg_prompts[i] if i < len(neg_prompts) else neg_prompts[-1]
            
            # [New] Extract LoRAs (Dynamic Loading)
            clean_pos, lora_list = self._parse_and_extract_loras(cur_pos)
            self.model_manager.manage_lora(lora_list, action="load")

            # Check [SKIP]
            if re.match(r"^\s*\[SKIP\]\s*$", cur_pos, re.IGNORECASE):
                self.log(f"  Skipping detection {i+1}: [SKIP] token found.")
                continue

            clean_pos = clean_pos.replace("[PROMPT]", "")
            cur_neg = cur_neg.replace("[PROMPT]", "")
            
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
            
            # Landmarks (needed for both Merge and Single modes if anatomy check is on)
            kps = None
            
            # [New] Use Pose KPS Override if available
            if pose_kps_override is not None:
                kps = pose_kps_override
                pass 
                
            elif config.get('auto_rotate') or config.get('anatomy_check'):
                kps = self.detector.get_face_landmarks(image, box)

            if config.get('anatomy_check') and kps is not None:
                if not is_anatomically_correct(kps):
                    self.log(f"  Skipping detection: Anatomically incorrect.")
                    continue

            # [Merge Mode Logic]
            if merge_mode != "None":
                # Collect Mask
                if merged_mask is None:
                    merged_mask = mask.copy()
                else:
                    merged_mask = cv2.bitwise_or(merged_mask, mask)
                
                # We need context for the merged run. We'll use the FIRST valid detection's config/box as base.
                # Or meaningful union. For now, store candidates.
                merge_candidates.append({
                    'config': det_config,
                    'box': box,
                    'kps': kps # Landmarks of individual faces are relevant for logic?
                               # Actually, merge mode usually disables per-face alignment/rotation because we process full image or large crop.
                               # But if we merge multiple faces, we can't align all of them at once unless we do full image inpaint.
                })
                continue # Skip individual processing

            # [Single Mode Logic] -> Run Inpaint immediately
            if self.preview_callback:
                # [Fix] Pass 'box' to ensure the detection box is visible during mask preview
                score_val = det.get('conf')
                score_text = f"{score_val:.2f}" if score_val is not None else None
                mask_vis = draw_mask_on_image(final_img, mask, color=(0, 255, 0), box=box, text=score_text)
                self.preview_callback(mask_vis)
            
            # [Delayed Gender Check]
            # Perform gender check here on the *aligned* crop (proc_img) if available.
            # This is much more accurate for rotated faces than checking the raw box.
            target_gender = config.get('gender_filter', 'All')
            if target_gender != 'All':
                 # We need to run gender check on 'proc_img' which is arguably the upright face.
                 # But detector needs context? No, InsightFace works on crops.
                 # We just need to convert 'proc_img' back to RGB or use it.
                 # Also, we might have skipped alignment if kps was None.
                 
                 # Logic:
                 # 1. Body Ratio (Already robust to rotation via Euclidean) -> Checked early? 
                 #    Actually, we can check Body Ratio early because it uses Pose model which is robust.
                 #    But InsightFace fails on rotation. So we move InsightFace check HERE.
                 
                 # Recalculate if Face Check is needed
                 detected_gender = None
                 
                 # Try InsightFace on Upright Crop
                 try:
                     # InsightFace expects full image or crop.
                     # It detects faces in the image. Since proc_img is tight crop, it should find 1 face.
                     face_gender = self.detector.analyze_gender(proc_img, None) # Pass None box to scan full crop
                     if face_gender:
                         detected_gender = face_gender
                 except:
                     pass
                     
                 # Fallback to Body Ratio (Global analysis on original image)
                 # We already have box for original image.
                 if detected_gender is None:
                      detected_gender = self.detector.analyze_body_gender(image, box)
                      if detected_gender:
                          self.log(f"    [Info] Face gender failed. Using Body analysis -> {detected_gender}")
                          
                 if detected_gender and detected_gender != target_gender:
                      self.log(f"    Skipping detection: Gender mismatch ({detected_gender} != {target_gender})")
                      continue

            denoise = self._calc_dynamic_denoise(box, (h, w), config['denoising_strength'])
            final_img = self._run_inpaint(final_img, mask, det_config, denoise, box, kps)

            if self.preview_callback:
                self.preview_callback(final_img)

        # [Execute Merge Pass]
        if merge_mode != "None" and merged_mask is not None and merge_candidates:
            self.log(f"  [Merge Mode] Processing merged mask ({len(merge_candidates)} objects, Mode={merge_mode}).")
            
            if merge_mode == "Merge and Invert":
                merged_mask = cv2.bitwise_not(merged_mask)

            # For merged inpainting, we typically use the Full Image or the bounding box of the merged mask.
            # ADetailer standard behavior for "Merge" is often running on the WHOLE image or a union bbox.
            # Here we will use the Full Image (box = full rect) to ensure everything is covered.
            # However, logic requires 'box'. Let's use image rect.
            full_box = [0, 0, w, h]
            
            # Config: Use the first candidate's prompt or the base config?
            # Usually base config prompt is safer for "Batch".
            # Let's use the first candidate's derived config (which has [SEP] logic applied).
            base_cand = merge_candidates[0]
            merged_config = base_cand['config'] 
            
            # Preview Merged Mask
            if self.preview_callback:
                mask_vis = draw_mask_on_image(final_img, merged_mask, color=(255, 255, 0)) # Cyan for merged
                self.preview_callback(mask_vis)

            # Run Inpaint (No Specific KPS for rotation, use full image)
            # Denoise: Use base strength, no dynamic adjustment based on box size because it's full image/complex mask
            denoise = config['denoising_strength']
            
            final_img = self._run_inpaint(final_img, merged_mask, merged_config, denoise, full_box, None)
            
            if self.preview_callback:
                self.preview_callback(final_img)
            
        return final_img

        return final_img

    def _run_inpaint(self, image, mask, config, strength, box, kps):
        # [Fix] Use Geometry Module for Alignment & Rotation
        padding_px = config['crop_padding']
        
        # Convert pixel padding to ratio for align_and_crop
        w_box, h_box = box[2] - box[0], box[3] - box[1]
        base_padding_ratio = (padding_px * 2) / max(w_box, h_box) if max(w_box, h_box) > 0 else 0.25
        
        # [BMAB Feature] Context Expansion
        # context_expand_factor: 1.0 by default. 
        # If 1.5, we add 0.5 to the padding ratio (effectively expanding the crop frame).
        expand_scale = config.get('context_expand_factor', 1.0)
        final_padding = base_padding_ratio + (expand_scale - 1.0)

        target_res = 512
        # [Fix] SDXL Model Detection for Default Resolution
        if hasattr(self.model_manager.pipe, "tokenizer_2"):
            target_res = 1024
            
        if config.get('inpaint_width', 0) > 0: target_res = config['inpaint_width']
        
        # [BMAB Feature] Safety Limit
        # Avoid OOM by capping resolution
        MAX_SAFE_RES = 2048
        if target_res > MAX_SAFE_RES:
            self.log(f"    [Warning] Target resolution {target_res} exceeds safety limit {MAX_SAFE_RES}. Capping.")
            target_res = MAX_SAFE_RES
        
        do_rotate = config.get('auto_rotate', False) and kps is not None
        
        # 1. Align & Crop (Image & Mask)
        # Pass final_padding instead of padding_ratio
        proc_img, M = align_and_crop(image, box, kps, target_size=target_res, padding=final_padding, force_rotate=do_rotate)
        proc_mask, _ = align_and_crop(mask, box, kps, target_size=target_res, padding=final_padding, force_rotate=do_rotate, borderMode=cv2.BORDER_CONSTANT)
        
        # [Fix] 합성 시 바운딩 박스 흔적을 없애기 위해 Soft Mask 보존
        paste_mask = proc_mask.copy()

        # Binarize mask after warping
        _, proc_mask = cv2.threshold(proc_mask, 127, 255, cv2.THRESH_BINARY)
        
        # [New] BMAB Basic Functions (Image Pre-processing)
        # Apply transforms to PROCESSED CROP (proc_img) BEFORE Inpainting
        # Order: Contrast -> Brightness -> Sharpness -> ColorTemp -> Noise
        
        # 1. Contrast
        if config.get('bmab_contrast', 1.0) != 1.0:
            pil_temp = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
            enhancer = Image.ImageEnhance.Contrast(pil_temp)
            pil_temp = enhancer.enhance(config['bmab_contrast'])
            proc_img = cv2.cvtColor(np.array(pil_temp), cv2.COLOR_RGB2BGR)

        # 2. Brightness
        if config.get('bmab_brightness', 1.0) != 1.0:
            pil_temp = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
            enhancer = Image.ImageEnhance.Brightness(pil_temp)
            pil_temp = enhancer.enhance(config['bmab_brightness'])
            proc_img = cv2.cvtColor(np.array(pil_temp), cv2.COLOR_RGB2BGR)

        # 3. Sharpness
        if config.get('bmab_sharpness', 1.0) != 1.0:
            pil_temp = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
            enhancer = Image.ImageEnhance.Sharpness(pil_temp)
            pil_temp = enhancer.enhance(config['bmab_sharpness'])
            proc_img = cv2.cvtColor(np.array(pil_temp), cv2.COLOR_RGB2BGR)

        # 4. Color Temperature
        if config.get('bmab_color_temp', 0.0) != 0.0:
            # Simple Tint approach: 
            # Positive -> Warmer (More Red/Yellow), Negative -> Cooler (More Blue)
            # In BGR: Warmer = Increase B (Blue) DOWN, Increase R/G UP?
            # Actually: Warm = +R, +G(slightly), -B. Cool = +B, +G(slightly), -R.
            temp_shift = config['bmab_color_temp']
            proc_img = proc_img.astype(np.float32)
            if temp_shift > 0: # Warm
                proc_img[:, :, 2] += temp_shift # R
                proc_img[:, :, 1] += temp_shift * 0.4 # G
                proc_img[:, :, 0] -= temp_shift # B
            else: # Cool
                proc_img[:, :, 0] += abs(temp_shift) # B
                proc_img[:, :, 1] += abs(temp_shift) * 0.4 # G
                proc_img[:, :, 2] -= abs(temp_shift) # R
            proc_img = np.clip(proc_img, 0, 255).astype(np.uint8)

        # 5. Noise Alpha (Pre-Inpaint Noise)
        # This differs from "Noise Mask" (which REPLACES mask area).
        # This ADDS noise to the input image for the AI to "texture" over.
        if config.get('bmab_noise_alpha', 0.0) > 0.0:
            alpha = config['bmab_noise_alpha']
            noise = np.random.normal(0, 50, proc_img.shape).astype(np.float32) # Sigma 50
            proc_img_f = proc_img.astype(np.float32)
            # proc_img = (1-alpha)*proc_img + alpha*noise ? No, standard blending.
            # Usually: Image + Noise * Strength
            proc_img_f = proc_img_f + (noise * alpha * 2.0) # Boost effect
            proc_img = np.clip(proc_img_f, 0, 255).astype(np.uint8)

        # 6. Edge Enhancement (Canny Sharpening)
        if config.get('bmab_edge_strength', 0.0) > 0.0:
            low = config.get('bmab_edge_low', 50)
            high = config.get('bmab_edge_high', 200)
            strength = config['bmab_edge_strength']
            
            # Detect Edges
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, low, high)
            
            # Blend Edges into Image (Darken edges)
            # Create an edge mask (inverted: white background, black edges)
            edges_inv = cv2.bitwise_not(edges)
            edges_bgr = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
            
            # Blend: proc_img * (1 - strength) + (proc_img * edges_bgr) * strength ???
            # Simpler: just darken the pixels where edges exist?
            # Or use PIL ImageEnhance.Sharpness logic? No, this is explicit Canny-based.
            
            # Method: weighted add? 
            # If edge pixel is white (255) in 'edges', we want to emphasize it in 'proc_img'.
            # Let's sharpen by adding edges to intensity channel? Or subtracting from it?
            # Typically "Edge Enhancement" means making edges more visible.
            
            # Simple approach: Overlay edges (black) onto image with alpha.
            # Convert edges to colored mask? No, just darken.
            proc_img_f = proc_img.astype(np.float32)
            mask = edges > 0
            
            # Darken edges by subtraction
            proc_img_f[mask] -= (255.0 * strength)
            proc_img = np.clip(proc_img_f, 0, 255).astype(np.uint8)
            
        new_h, new_w = proc_img.shape[:2]
        
        pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(proc_mask)

        # [New] Noise Mask Implementation (Renamed/Legacy)
        # This replaces the MASKED area entirely with noise.
        # BMAB Basic Noise Alpha above modifies the INPUT picture globally (or inside crop).
        # We keep both features.
        if config.get('use_noise_mask', False):
             # Create Gaussian Noise
             # [Fix] Generate 3-channel noise directly or convert properly
             # noise = np.random.normal(0, 1, proc_img.shape).astype(np.uint8) * 255 -> This creates (H,W,3) but low values
             # Better: Random uniform noise [0, 255]
             noise = np.random.randint(0, 256, proc_img.shape, dtype=np.uint8)
             # No need for cvtColor if we generated (H, W, 3) directly from shape
             
             # Blend Noise into Image ONLY at Masked Area
             # ADetailer standard: Replace original content with noise in the masked area
             # But usually Stable Diffusion needs some structure. ADetailer typically does 'fill' latent.
             # Since we are pixel based here, we can blend noise.
             # Stronger approach: Replace masked area with noise completely? 
             # Let's trust the user wants "Noise" (high denoising)
             
             # Efficient numpy masking
             mask_indices = proc_mask > 127
             proc_img_noise = proc_img.copy()
             
             # Random noise (0-255)
             noise_layer = np.random.randint(0, 256, proc_img.shape, dtype=np.uint8)
             
             proc_img_noise[mask_indices] = noise_layer[mask_indices]
             pil_img = Image.fromarray(cv2.cvtColor(proc_img_noise, cv2.COLOR_BGR2RGB))
             self.log(f"    [Info] Applied Noise Mask.")

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
                else:
                    # [New] Advanced ControlNet Preprocessor Selection
                    # Determine Preprocessor based on Model Name AND 'control_module' config
                    cn_module = config.get('control_module', 'inpaint_global_harmonious')
                    
                    # Auto-detect if module is default/None but model implies specific type
                    if cn_module in ["None", "inpaint_global_harmonious"] and "canny" not in cn_model:
                        if "depth" in cn_model: cn_module = "depth_midas"
                        elif "scribble" in cn_model: cn_module = "scribble_pidinet"
                        elif "lineart" in cn_model: cn_module = "lineart_realistic"
                        elif "openpose" in cn_model: cn_module = "openpose_full"
                        elif "canny" in cn_model: cn_module = "canny"
                    
                    # Apply Preprocessing
                    try:
                        if cn_module == "openpose_full":
                           from controlnet_aux import OpenposeDetector
                           processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                           processor.to(self.device)
                           control_args["control_image"] = processor(pil_img)
                           
                        elif cn_module == "depth_midas":
                           from controlnet_aux import MidasDetector
                           processor = MidasDetector.from_pretrained("lllyasviel/ControlNet")
                           processor.to(self.device)
                           control_args["control_image"] = processor(pil_img)

                        elif cn_module == "lineart_realistic":
                           from controlnet_aux import LineartDetector
                           processor = LineartDetector.from_pretrained("lllyasviel/ControlNet")
                           processor.to(self.device)
                           control_args["control_image"] = processor(pil_img, coarse=True)
                           
                        elif cn_module == "scribble_pidinet":
                           from controlnet_aux import PidiNetDetector
                           processor = PidiNetDetector.from_pretrained("lllyasviel/ControlNet")
                           processor.to(self.device)
                           control_args["control_image"] = processor(pil_img, safe=True)
                           
                        elif cn_module == "canny":
                           # OpenCV Canny (Lightweight)
                           canny = cv2.Canny(proc_img, 100, 200)
                           canny = np.stack([canny] * 3, axis=-1)
                           control_args["control_image"] = Image.fromarray(canny)
                           
                        else:
                           # Fallback / Inpaint Global Harmonious (Just Input Image)
                           control_args["control_image"] = pil_img
                           
                    except ImportError:
                        self.log(f"    [Warning] controlnet_aux missing for {cn_module}. Using input image.")
                        control_args["control_image"] = pil_img
                    except Exception as e:
                        self.log(f"    [Warning] Preprocessor {cn_module} failed: {e}. Using input image.")
                        control_args["control_image"] = pil_img

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
            
            # [Fix] NaN / Black Image Detection & Auto-Recovery
            # 추론 결과가 검은색(0)이거나 NaN이 포함되어 있으면 실패로 간주하고 모델을 재로드함
            res_np_check = np.array(output)
            if np.isnan(res_np_check).any() or (res_np_check.max() == 0 and res_np_check.min() == 0):
                self.log(f"    [Critical] NaN/Black image detected. Forcing model reload.")
                self.model_manager.pipe = None # 파이프라인 강제 폐기
                gc.collect()
                torch.cuda.empty_cache()
                # 빈 이미지 반환하여 프로그램이 죽지 않게 함
                return image

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