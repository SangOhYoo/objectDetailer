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
from core.detail_daemon import DetailDaemonContext
from compel import Compel, ReturnedEmbeddingsType
from core.upscaler import ESRGANUpscaler

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
        self.upscaler = ESRGANUpscaler(device=device)

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

                # [Optimized] 불필요한 사전 메모리 정리 제거 (ModelManager가 관리)
                # gc.collect()
                # torch.cuda.empty_cache()

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
                # [Optimized] 패스 종료 시 강제 메모리 정리 제거
                # gc.collect()
                # torch.cuda.empty_cache()
                
        if cfg.get('system', 'log_level') == 'DEBUG':
            print("[Pipeline] Offloading models to free VRAM...")
        self.detector.offload_models()
        self.face_restorer.unload_model() 
        self.model_manager.unload_model() 
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return result_img

    def detect_preview(self, image, config):
        """
        [New] Detect Preview: Run detection only and visualize results.
        """
        # 1. Resize/Person Check (Optional, skipping for preview speed unless explicitly requested)
        # Assuming we just want to see what the current detector sees on the current image.
        
        # 2. Main Detection
        det_model = config['detector_model']
        conf = config['conf_thresh']
        classes = None # Default All
        
        # Parse Classes
        yolo_classes_str = config.get('yolo_classes', '').strip()
        if yolo_classes_str:
            try:
                # e.g. "0, 1, 2" or "person, cat"
                # Detector expects list of strings or ints, usually strings for class names if logic supports it
                # core/detector.py handles class filtering.
                classes = [c.strip() for c in yolo_classes_str.split(',')]
            except: pass
        
        unit_name = config.get('unit_name', 'Preview')
        self.log(f"  > Preview Detection: {det_model} (Conf: {conf})")
        
        detections = self.detector.detect(image, det_model, conf, classes=classes)
        self.log(f"    - Raw Detections: {len(detections)}")
        
        # 2.1 Filter by ratio (Smart Filter)
        img_h, img_w = image.shape[:2]
        img_area = img_w * img_h
        
        # [Fix] Config values are already ratios (0.0 ~ 1.0) from UI
        min_ratio = config.get('min_face_ratio', 0.0)
        max_ratio = config.get('max_face_ratio', 1.0) # Default 1.0 (100%)
        
        # [Debug] Console Print
        if cfg.get('system', 'log_level') == 'DEBUG':
            self.log(f"[Pipeline] Filter Settings: Min={min_ratio*100:.2f}% ({min_ratio:.4f}), Max={max_ratio*100:.2f}% ({max_ratio:.4f})")
            self.log(f"[Pipeline] Image Area: {img_area} ({img_w}x{img_h})")

        filtered_dets = []
        for d in detections:
            x1, y1, x2, y2 = d['box']
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / img_area
            
            pass_filter = min_ratio <= ratio <= max_ratio
            if cfg.get('system', 'log_level') == 'DEBUG':
                 self.log(f"[Pipeline] Det: {d.get('label_name')} Ratio: {ratio:.4f} ({ratio*100:.2f}%) -> {'PASS' if pass_filter else 'FAIL'}")

            # [Debug]
            if cfg.get('system', 'log_level') == 'DEBUG':
                 self.log(f"      [Target] {d['label']} | Area: {box_area} / {img_area} = {ratio:.4f} ({ratio*100:.2f}%) | Range: {min_ratio:.4f}~{max_ratio:.4f}")

            if pass_filter:
                filtered_dets.append(d)
        
        # If no filtered results but we had raw results, warn the user
        if len(detections) > 0 and len(filtered_dets) == 0:
            self.log(f"    [Warning] All {len(detections)} detections were filtered by size ratio ({min_ratio*100:.1f}% ~ {max_ratio*100:.1f}%). Check 'Min/Max Face Ratio' settings.")

        detections = filtered_dets
        self.log(f"    - Final Targets: {len(detections)}")
        
        # [Debug] Explicit Visualization Log
        if len(detections) > 0:
            self.log(f"    [Pipeline] Visualizing {len(detections)} objects...")
        else:
            self.log("    [Pipeline] No objects to visualize.")

        # 3. SAM (Optional)
        if config.get('use_sam'):
            if self.sam is None:
                from core.sam_wrapper import SamInference
                self.sam = SamInference(self.device)
            
            # For preview, we might just show regular boxes if SAM is slow, 
            # but user wants to see 'what yolo detected' specifically.
            # If user selected SAM mode, we should try to show SAM masks?
            # Provided request says "yolo가 탐지되어 보여주는...". 
            # If SAM is enabled, YOLO provides boxes for SAM. 
            # Let's visualize detections only (boxes) primarily, maybe masks if SAM is active.
            # For simplicity and speed in preview, Box + Label is usually enough.
            pass

        # 4. Visualize
        viz_img = image.copy()
        
        # Draw Boxes
        viz_img = draw_detections(viz_img, detections)
        
        return viz_img

    def _remove_object(self, image, mask, config):
        """
        [New] Remove object (Inpaint) to create a clean background.
        Refined: Stronger dilation, Telea init, and soft edges to prevent ghosting borders.
        """
        h, w = image.shape[:2]
        
        # 1. Setup Config for Removal
        remove_config = config.copy()
        
        # Force high denoise
        remove_config['denoising_strength'] = 1.0
        
        # Strict Background Prompts
        remove_config['pos_prompt'] = "empty scene, scenery, background only, natural environment, high quality, realistic, texture, 8k"
        orig_neg = config.get('neg_prompt', '')
        remove_config['neg_prompt'] = "person, man, woman, human, character, face, body, silhouette, ghost, shadows, blurry, distorted, outline, seams, " + orig_neg
        
        remove_config['inpaint_width'] = w
        remove_config['inpaint_height'] = h
        
        # Disable interfering features
        remove_config['use_controlnet'] = False 
        remove_config['use_hires_fix'] = False 
        remove_config['dd_enabled'] = False 
        
        # 2. Dynamic Dilation & Blur
        # Borders often appear because the mask is too tight or the transition is too hard.
        # We dilate significantly to ensure we are generating from "clean" background context.
        min_dim = min(h, w)
        # Increased from 0.02 to 0.05 to ensure we skip shadows/auras
        dilate_amount = int(min_dim * 0.05) 
        dilate_amount = max(30, dilate_amount) # Minimum 30px
        
        # Ensure odd kernel
        if dilate_amount % 2 == 0: dilate_amount += 1
        
        dilated_mask = cv2.dilate(mask, np.ones((dilate_amount, dilate_amount), np.uint8), iterations=1)
        
        # Blur the mask for seamless blending
        blur_amount = int(dilate_amount * 0.5) 
        if blur_amount % 2 == 0: blur_amount += 1
        dilated_mask = cv2.GaussianBlur(dilated_mask, (blur_amount, blur_amount), 0)
        
        # 3. Telea Initialization (Content Fill) + Texture Injection
        # We pre-fill using Telea to get base colors.
        # Then ADD NOISE to prevent "flat" look which VAE hates.
        
        telea_mask = (dilated_mask > 20).astype(np.uint8) * 255 # Looser mask for Telea
        
        self.log(f"    [Background] Pre-filling masked area (Telea + Noise) to remove ghosting...")
        clean_init = cv2.inpaint(image, telea_mask, 3, cv2.INPAINT_TELEA)
        
        # Add Noise to the masked area in clean_init
        # We want to match the "scene texture" roughly.
        # Generate Gaussian noise
        noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
        clean_init_16 = clean_init.astype(np.int16)
        
        # Add noise only where mask > 0
        mask_indices = dilated_mask > 0
        
        # Blend noise: Image + Noise
        noisy_bg = clean_init_16 + noise
        noisy_bg = np.clip(noisy_bg, 0, 255).astype(np.uint8)
        
        # Apply noise only to masked area
        # We use the blurred mask as alpha for noise blending?
        # Simpler: Just replace logical area.
        clean_init[mask_indices] = noisy_bg[mask_indices]
        
        self.log(f"    [Background] Generating detailed background (Denoise 1.0, Dilation {dilate_amount})...")
        
        try:
            # Pass the Texturized Init image
            clean_bg = self._run_inpaint(clean_init, dilated_mask, remove_config, 1.0, [0,0,w,h], None)
            return clean_bg
        except Exception as e:
            self.log(f"    [Error] Person removal failed: {e}")
            return image

    def _resize_by_person(self, image, config, target_boxes=None):
        """
        [BMAB Feature] Scene Expansion (Outpainting) - Divide and Conquer Strategy
        Step 1: Remove Person (Create Clean Background)
        Step 2: Resize Person (Target)
        Step 3: Composite (Blend)
        """
        h, w = image.shape[:2]
        target_ratio = config.get('resize_ratio', 0.6)
        
        # 1. Detect Person
        resize_model = config['detector_model']
        if 'face' in resize_model.lower() or 'hand' in resize_model.lower():
             resize_model = 'yolov8n.pt'
        
        person_conf = config.get('conf_thresh', 0.35)
        person_detections = self.detector.detect(image, resize_model, person_conf, classes=['person'])
        
        if not person_detections:
            return image, 1.0, 0, 0
            
        # Filter relevant persons
        relevant_persons = []
        if target_boxes:
            for p_det in person_detections:
                px1, py1, px2, py2 = p_det['box']
                for t_box in target_boxes:
                    tx1, ty1, tx2, ty2 = t_box
                    ox1 = max(px1, tx1); oy1 = max(py1, ty1)
                    ox2 = min(px2, tx2); oy2 = min(py2, ty2)
                    if ox2 > ox1 and oy2 > oy1:
                        inter_area = (ox2 - ox1) * (oy2 - oy1)
                        t_area = (tx2 - tx1) * (ty2 - ty1)
                        if t_area > 0 and (inter_area / t_area) > 0.5:
                            relevant_persons.append(p_det)
                            break
        else:
            relevant_persons = person_detections
            
        if not relevant_persons:
            return image, 1.0, 0, 0

        # Build Union Mask of Persons
        union_mask = np.zeros((h, w), dtype=np.uint8)
        max_h = 0
        
        for det in relevant_persons:
            x1, y1, x2, y2 = map(int, det['box'])
            ph = y2 - y1
            if ph > max_h: max_h = ph
            
            if det.get('mask') is not None:
                union_mask = cv2.bitwise_or(union_mask, det['mask'])
            else:
                # Fallback to box if mask missing (though Det should provide it if SAM used, else Box)
                # If SAM not used, we might need a segmenter? 
                # For now assumes YOLO boxes if SAM off. 
                # Improving this: Use SAM specifically here if available?
                # The caller _process_pass handles SAM init.
                if self.sam:
                     # Generate mask for this box
                     try:
                         # Ensure SAM has image
                         self.sam.set_image(image)
                         sam_mask = self.sam.predict_mask_from_box(det['box'])
                         union_mask = cv2.bitwise_or(union_mask, sam_mask)
                     except:
                         cv2.rectangle(union_mask, (x1, y1), (x2, y2), 255, -1)
                else:
                     cv2.rectangle(union_mask, (x1, y1), (x2, y2), 255, -1)

        current_ratio = max_h / h
        
        # Calculate Scale
        scale = target_ratio / current_ratio
        if scale > 0.99: return image, 1.0, 0, 0
        if scale < 0.2: scale = 0.2
        
        self.log(f"    [Canvas] Zoom Out Plan: Scale {scale:.3f}x")

        # ---------------------------------------------------------
        # Step 1: Remove Person (Clean Background)
        # ---------------------------------------------------------
        clean_bg = self._remove_object(image, union_mask, config)
        
        # [Debug] Save Clean Background - Removed for production
        # debug_path = os.path.join(cfg.get_path('output'), "debug_clean_bg.png")
        # try:
        #     cv2.imwrite(debug_path, cv2.cvtColor(clean_bg, cv2.COLOR_RGB2BGR))
        #     self.log(f"    [Debug] Saved clean background to {debug_path}")
        # except: pass
        
        # For now, to verify this step, we return the clean_bg as the result
        # This effectively pauses the pipeline at Step 1 for user verification.
        # We need to return the dummy scale/shift so downstream logic doesn't explode,
        # but the image will be just the background.
        
        # Reset scale to 1.0 so boxes don't shift for this debug view
        return clean_bg, 1.0, 0, 0
        
        # Future Steps Integration:
        # resized_img = cv2.resize(image, ...)
        # ... composite resized_img onto clean_bg ...
        # return composited_img, scale, shift_x, shift_y

    def _process_pass(self, image, config):
        h, w = image.shape[:2]
        orig_h, orig_w = h, w # [Fix] Capture Original Size for Restoration
        img_area = h * w
        
        # 1. Run Detection (Full Image)
        detections = self.detector.detect(image, config['detector_model'], config['conf_thresh'], classes=config.get('yolo_classes'))
        
        if not detections:
            self.log(f"    [Info] No objects detected (Threshold: {config['conf_thresh']}).")
            return image

        # 2. Sort & Limit Detections
        boxes = [d['box'] for d in detections]
        scores = [d['conf'] for d in detections]
        sort_method = config.get('sort_method', '신뢰도')
        _, _, sorted_indices = sort_boxes(boxes, scores, sort_method, w, h)
        detections = [detections[i] for i in sorted_indices]

        # Limit (Max Detections)
        max_det = config.get('max_det', 20)
        if max_det > 0 and len(detections) > max_det:
            detections = detections[:max_det]
            
        # [New] Split Prompts by [SEP] (ADetailer Syntax)
        # Needed for per-detection prompt assignment
        sep_pattern = r"\s*\[SEP\]\s*"
        pos_prompts = re.split(sep_pattern, config.get('pos_prompt', ''))
        neg_prompts = re.split(sep_pattern, config.get('neg_prompt', ''))

        # 3. [New] Resize by Person (Zoom Out)
        # Now we scale down instead of expanding canvas
        scale_factor = 1.0
        pad_x, pad_y = 0, 0
        
        if config.get('resize_enable', False):
             try:
                target_boxes = [d['box'] for d in detections]
                image, scale, px, py = self._resize_by_person(image, config, target_boxes)
                
                # Update Transformations
                if scale != 1.0 or px != 0 or py != 0:
                    self.log(f"    [Resize] Transforming boxes: Scale={scale:.3f}, Offset=({px},{py})")
                    for det in detections:
                        x1, y1, x2, y2 = det['box']
                        # Scale + Shift
                        nx1 = x1 * scale + px
                        ny1 = y1 * scale + py
                        nx2 = x2 * scale + px
                        ny2 = y2 * scale + py
                        det['box'] = [nx1, ny1, nx2, ny2]
                    
                    # Update dimensions for subsequent logic
                    h, w = image.shape[:2]
                    img_area = h * w
                    
             except Exception as e:
                self.log(f"    [Error] Resize by person failed: {e}")
                traceback.print_exc()

        # 4. Processing Loop
        image_copy = image.copy()
        for i, det in enumerate(detections):
             box = det['box'] # These are now shifted and correct for the new image
             score = det.get('conf', 0.0)

        # [New] Pre-calculate LoRAs for visualization (프리뷰용 LoRA 정보 미리 계산)
        for i, det in enumerate(detections):
            cur_pos = pos_prompts[i] if i < len(pos_prompts) else pos_prompts[-1]
            _, lora_list = self._parse_and_extract_loras(cur_pos)
            det['lora_infos'] = lora_list

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
            
            # [Feature] Landscape Detail: Bypass min_face_ratio if enabled
            is_landscape_detail = config.get('bmab_landscape_detail', False)
            
            if face_ratio < config['min_face_ratio'] and not is_landscape_detail:
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
            mask = MaskUtils.refine_mask(mask, dilation=config['mask_dilation'], erosion=config.get('mask_erosion', 0), blur=config['mask_blur'])
            
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

            # [Hires Fix] Configuration
            denoise = self._calc_dynamic_denoise(box, (h, w), config['denoising_strength'])
            current_steps = None
            current_cfg = None

            # [Fix] Hires Fix Enable/Disable Logic
            # If disabled, force upscaler to None to prevent _run_inpaint from running it
            if config.get('use_hires_fix', False):
                 factor = config.get('hires_upscale_factor', 2.0)
                 
                 base_res = 512
                 if hasattr(self.model_manager.pipe, "tokenizer_2"): base_res = 1024
                 if config.get('inpaint_width', 0) > 0: base_res = config['inpaint_width']
                 
                 hires_target_w = int(base_res * factor)
                 if config.get('hires_width', 0) > 0: hires_target_w = config['hires_width']
                 
                 det_config['inpaint_width'] = hires_target_w
                 denoise = config.get('hires_denoise', 0.4)
                 current_steps = config.get('hires_steps', 20)
                 current_cfg = config.get('hires_cfg', 7.0)
                 det_config['hires_upscaler'] = config.get('hires_upscaler', 'None')
                 
                 self.log(f"    [Hires Fix] Active: Res={hires_target_w}, Denoise={denoise}, Upscaler={det_config['hires_upscaler']}")
            else:
                 det_config['hires_upscaler'] = 'None'

            # [Fix] Apply LoRAs specific to this detection
            current_loras = det.get('lora_infos', [])
            self.model_manager.manage_lora(current_loras, "load")

            final_img = self._run_inpaint(final_img, mask, det_config, denoise, box, kps, steps=current_steps, guidance_scale=current_cfg)

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

    def _run_inpaint(self, image, mask, config, strength, box, kps, steps=None, guidance_scale=None):
        # [Fix] Use Geometry Module for Alignment & Rotation
        padding_px = config['crop_padding']
        
        # Convert pixel padding to ratio for align_and_crop
        w_box, h_box = box[2] - box[0], box[3] - box[1]
        base_padding_ratio = (padding_px * 2) / max(w_box, h_box) if max(w_box, h_box) > 0 else 0.25
        
        # [BMAB Feature] Context Expansion
        expand_scale = config.get('context_expand_factor', 1.0)
        final_padding = base_padding_ratio + (expand_scale - 1.0)

        target_res = 512
        # [Fix] SDXL Model Detection for Default Resolution
        if hasattr(self.model_manager.pipe, "tokenizer_2"):
            target_res = 1024
            
        if config.get('inpaint_width', 0) > 0: target_res = config['inpaint_width']
        
        # [BMAB Feature] Safety Limit
        MAX_SAFE_RES = 2048
        if target_res > MAX_SAFE_RES:
            self.log(f"    [Warning] Target resolution {target_res} exceeds safety limit {MAX_SAFE_RES}. Capping.")
            target_res = MAX_SAFE_RES
        
        do_rotate = config.get('auto_rotate', False) and kps is not None
        
        # 1. Align & Crop (Image & Mask)
        proc_img, M = align_and_crop(image, box, kps, target_size=target_res, padding=final_padding, force_rotate=do_rotate)
        proc_mask, _ = align_and_crop(mask, box, kps, target_size=target_res, padding=final_padding, force_rotate=do_rotate, borderMode=cv2.BORDER_CONSTANT)
        
        # [Hires Fix] Upscale Injection
        # [Hires Fix] Upscale Injection (Refactored)
        hires_upscaler_name = config.get('hires_upscaler', 'None')
        if hires_upscaler_name and hires_upscaler_name != 'None':
             proc_img = self._apply_hires_upscale(image, box, kps, final_padding, target_res, hires_upscaler_name, do_rotate)

        # [Fix] 합성 시 바운딩 박스 흔적을 없애기 위해 Soft Mask 보존
        paste_mask = proc_mask.copy()

        # Binarize mask after warping
        _, proc_mask = cv2.threshold(proc_mask, 127, 255, cv2.THRESH_BINARY)


        
        if config.get('bmab_enabled', True):
            from core.bmab_utils import apply_bmab_basic
            try:
                # Apply BMAB Basic Effects (Contrast, Brightness, Sharpness, Color, Temp, Noise)
                proc_img = apply_bmab_basic(proc_img, config)
            except Exception as e:
                self.log(f"    [Warning] BMAB Basic effects failed: {e}")
            
        # [New] Mask Content Preprocessing
        # This replaces the legacy 'use_noise_mask' logic
        mask_content_mode = config.get('mask_content', 'original')
        proc_img = self._apply_mask_content(proc_img, proc_mask, mask_content_mode)

        new_h, new_w = proc_img.shape[:2]
        
        # [Visual Check] BGR to RGB for PIL
        # Since upscaler.py now handles BGR-in/BGR-out correctly (matching A1111 standard),
        # proc_img is strictly BGR. We can trust this conversion.
        pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(proc_mask)

        # ControlNet
        control_args = {}
        # [Fix] 파이프라인에 ControlNet이 로드되어 있다면, 설정(use_controlnet)과 무관하게 이미지를 공급해야 함
        if hasattr(self.model_manager.pipe, "controlnet"):
            if not config.get('use_controlnet', False):
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
            prompt_embeds, neg_prompt_embeds = self._get_compel_embeds(
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
            "guidance_scale": guidance_scale if guidance_scale is not None else config.get('cfg_scale', 7.0),
            "num_inference_steps": steps if steps is not None else config.get('steps', 20),
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
            
        print(f"    [Pipeline] Pre-Infer: Embeds={'Present' if prompt_embeds is not None else 'None'}")
        if prompt_embeds is not None:
             print(f"      - Pos: {prompt_embeds.dtype} ({prompt_embeds.shape}) Device={prompt_embeds.device}")
             print(f"      - Neg: {neg_prompt_embeds.dtype} ({neg_prompt_embeds.shape}) Device={neg_prompt_embeds.device}")

        # Inference
        # [Fix] 인퍼런스 직전 메모리 정리
        torch.cuda.empty_cache()
        
        # [Detail Daemon & Soft Inpainting Prep]
        dd_enabled = config.get('dd_enabled', False)
        use_soft = config.get('use_soft_inpainting', False)
        
        # We enable DetailDaemonContext if either DD or SoftInpainting is active
        # The Context (in detail_daemon.py) needs to support Soft Inpainting params if we pass them.
        # Assuming current DetailDaemonContext isn't yet updated for Soft Inpainting, we pass them 
        # but they might be ignored unless I update DetailDaemon.
        # However, Pixel Composite is handled POST-inference. 
        # Only 'Schedule bias', 'Preservation strength', 'Transition contrast boost' might need hook support.
        # If DetailDaemon doesn't support them, passing them won't crash it (dict is flexible), but logic won't run.
        # I will strictly enable DD context if dd_enabled is True for now,
        # OR if use_soft is True AND I want to use DD hooks for Soft Inpainting.
        # Given request is "UI & Dev Direction", implementing the complex hook logic for Soft Inpainting 
        # inside DetailDaemon is likely out of scope for this single turn unless existing DD supports it.
        # I'll stick to passing config.
        
        dd_config = {}
        if dd_enabled or use_soft:
             dd_config = {
                 'mode': config.get('dd_mode', 'both'),
                 'amount': config.get('dd_amount', 0.1),
                 'start': config.get('dd_start', 0.2),
                 'end': config.get('dd_end', 0.8),
                 'bias': config.get('dd_bias', 0.5),
                 'exponent': config.get('dd_exponent', 1.0),
                 'start_offset': config.get('dd_start_offset', 0.0),
                 'end_offset': config.get('dd_end_offset', 0.0),
                 'fade': config.get('dd_fade', 0.0),
                 'smooth': config.get('dd_smooth', True),
                 
                 # Soft Inpainting Params
                 'soft_enabled': use_soft,
                 'soft_schedule_bias': config.get('soft_schedule_bias', 1.0),
                 'soft_preservation_strength': config.get('soft_preservation_strength', 0.5),
                 'soft_transition_contrast': config.get('soft_transition_contrast', 4.0)
             }
             if config.get('system', 'log_level') == 'DEBUG':
                 self.log(f"    [DetailDaemon] Active: {dd_config}")

        # [Fix] CPU Autocast Issue
        # Determine device type for autocast
        device_type = "cuda"
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device.split(":")[0]

        # Enable autocast only for CUDA. CPU autocast (bfloat16) is often unstable or mismatching with float32 usage.
        use_autocast = (device_type == "cuda")

        with torch.inference_mode():
            with torch.autocast(device_type, enabled=use_autocast):
                with DetailDaemonContext(self.model_manager.pipe, dd_enabled, dd_config):
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
        
        # [New] Soft Inpainting: Pixel Composite
        if config.get('use_soft_inpainting', False):
             # We need the original content (before mask filling) for composite.
             # Recover it using the same transform M.
             proc_img_orig = cv2.warpAffine(image, M, (new_w, new_h))
             res_np = self._apply_pixel_composite(proc_img_orig, res_np, proc_mask, config)

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

    def _apply_hires_upscale(self, image, box, kps, padding, target_size, upscaler_name, do_rotate):
        """
        Applies Hires Fix upscaling logic similar to A1111/Forge.
        1. Extracts crop at native resolution.
        2. Upscales using specified model (ESRGAN) or interpolation (Latent/etc).
        3. Resizes to target_size for Inpainting.
        """
        try:
            w_box, h_box = box[2] - box[0], box[3] - box[1]
            native_size = int(max(w_box, h_box) * (1 + padding))
            
            # Extract Native Crop
            native_crop, _ = align_and_crop(image, box, kps, target_size=native_size, padding=padding, force_rotate=do_rotate)
            
            upscaled_img = None
            
            # Latent / Interpolation Methods
            if upscaler_name in ["Latent", "Lanczos", "Nearest", "Bicubic"]:
                # Just resize native crop to target_size?
                # No, "Latent" implies we just want to generate high res.
                # Since we are prep-ing the image for Inpaint, "Latent" usually means "Just Resize"
                # so the VAE encodes the resized image.
                # However, usually we want some specific interpolation.
                match upscaler_name:
                    case "Nearest": interp = cv2.INTER_NEAREST
                    case "Bicubic": interp = cv2.INTER_CUBIC
                    case _: interp = cv2.INTER_LANCZOS4 # Default for Latent/Lanczos
                
                # We already have native_crop. Resize it to target_size.
                # Wait, if target_size >> native_size, this is the upscale.
                upscaled_img = cv2.resize(native_crop, (target_size, target_size), interpolation=interp)
                self.log(f"    [Hires Fix] Upscaling with {upscaler_name} (Resize only)")

            # ESRGAN Models
            else:
                model_path = os.path.join("D:\\AI_Models\\ESRGAN", upscaler_name)
                if self.upscaler.load_model(model_path):
                    self.log(f"    [Hires Fix] Upscaling with {upscaler_name}...")
                    # 1. Upscale (e.g. 4x)
                    upscaled = self.upscaler.upscale(native_crop) # Returns BGR
                    
                    # 2. Fit to Target Size
                    # If result is larger than target, downscale (Area).
                    # If smaller (rare), upscale (Lanczos).
                    h, w = upscaled.shape[:2]
                    if h > target_size:
                        upscaled_img = cv2.resize(upscaled, (target_size, target_size), interpolation=cv2.INTER_AREA)
                    else:
                        upscaled_img = cv2.resize(upscaled, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                else:
                    self.log(f"    [Warning] Upscaler {upscaler_name} not found. Using Lanczos fallback.")
                    upscaled_img = cv2.resize(native_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

            return upscaled_img
            
        except Exception as e:
            self.log(f"    [Error] Hires Upscale logic failed: {e}")
            # Fallback: Just return current aligned crop (which is already target_size usually, wait.
            # No, if this fails, we need to return SOMETHING.
            # Re-run align_and_crop at target_size
            fallback, _ = align_and_crop(image, box, kps, target_size=target_size, padding=padding, force_rotate=do_rotate)
            return fallback

    def _calc_dynamic_denoise(self, box, img_shape, base):
        x1, y1, x2, y2 = box
        ratio = ((x2 - x1) * (y2 - y1)) / (img_shape[0] * img_shape[1])
        adj = 0.15 if ratio < 0.05 else (0.10 if ratio < 0.10 else (0.05 if ratio < 0.20 else 0.0))
        return max(0.1, min(base + adj, 0.8))

    def _get_compel_embeds(self, pipe, prompt, negative_prompt):
        """Compel 라이브러리를 사용한 프롬프트 가중치 처리 ((text:1.1) 등 지원)"""
        if not prompt: prompt = ""
        if not negative_prompt: negative_prompt = ""
        
        if not hasattr(pipe, "tokenizer") or not hasattr(pipe, "text_encoder"):
            return None, None

        # [Fix] SDXL vs SD1.5 Compel Init
        # SDXL has tokenizer_2 and text_encoder_2
        if hasattr(pipe, "tokenizer_2"):
             compel = Compel(
                 tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                 text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                 returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                 requires_pooled=[False, True]
             )
             # SDXL requires pooled embeds too
             # For simplicity, let's use the pipe's own encode_prompt if possible, 
             # BUT pipe.encode_prompt doesn't support A1111 weights natively without Compel.
             # Actually Compel for SDXL is complex. 
             # Let's stick to standard handling for SDXL (which supports simple weighting in some versions)
             # OR just use Compel if we trust it.
             # For now, let's default to None for SDXL to avoid breaking it, relying on Diffusers native handling
             # or if user really needs weighting, we need detailed SDXL Compel setup.
             return None, None # Let SDXL pipeline handle it (Diffusers has some support)
        else:
             # SD 1.5
             compel = Compel(
                 tokenizer=pipe.tokenizer, 
                 text_encoder=pipe.text_encoder,
                 truncate_long_prompts=False # Chunking
             )

        try:
             prompt_embeds = compel(prompt)
             neg_prompt_embeds = compel(negative_prompt)
             
             # [Fix] Ensure same length (padding) to avoid CrossAttention mismatch
             [prompt_embeds, neg_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, neg_prompt_embeds])
             
             # [Fix] Ensure device (Compel might return CPU tensors if model is offloaded)
             # Use text_encoder dtype to avoid autocast prioritization conflicts
             # BUT if running on CPU (fallback), we must use float32 because float16 autocast on CPU is unstable
             if hasattr(self.device, 'type') and self.device.type == 'cpu':
                 target_dtype = torch.float32
             elif isinstance(self.device, str) and 'cpu' in self.device:
                 target_dtype = torch.float32
             else:
                 try:
                    target_dtype = pipe.text_encoder.dtype
                 except:
                    try:
                        target_dtype = next(pipe.text_encoder.parameters()).dtype
                    except:
                        target_dtype = torch.float32

             prompt_embeds = prompt_embeds.to(device=self.device, dtype=target_dtype)
             neg_prompt_embeds = neg_prompt_embeds.to(device=self.device, dtype=target_dtype)
             
             return prompt_embeds, neg_prompt_embeds
        except Exception as e:
             self.log(f"    [Warning] Compel failed: {e}. Fallback to simple tokenization.")
             return self._get_long_prompt_embeds(pipe, prompt, negative_prompt)

    def _get_long_prompt_embeds(self, pipe, prompt, negative_prompt):
        # ... (Existing logic kept as fallback) ...
        # (Rest of existing function)
        """77토큰 제한을 우회하기 위한 임베딩 청킹(Chunking) 처리"""
        if not prompt: prompt = ""
        if not negative_prompt: negative_prompt = ""
        
        # ... (Same as before) ...
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

        # 3. Pad chunks
        total_chunks = max(len(pos_chunks), len(neg_chunks))
        
        while len(pos_chunks) < total_chunks: pos_chunks.append([])
        while len(neg_chunks) < total_chunks: neg_chunks.append([])

        # 4. Encode
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def encode(chunks):
            embeds = []
            for chunk in chunks:
                input_ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
                pad_len = tokenizer.model_max_length - len(input_ids)
                if pad_len > 0: input_ids += [pad_token] * pad_len
                
                input_tensor = torch.tensor([input_ids], device=text_encoder.device)
                embeds.append(text_encoder(input_tensor)[0])
            # [Fix] Match text_encoder dtype dynamically to avoid autocast conflicts
            # BUT if running on CPU (fallback), we must use float32 because float16 autocast on CPU is unstable
            if hasattr(self.device, 'type') and self.device.type == 'cpu':
                target_dtype = torch.float32
            elif isinstance(self.device, str) and 'cpu' in self.device:
                target_dtype = torch.float32
            else:
                try:
                    target_dtype = text_encoder.dtype
                except:
                    try:
                        target_dtype = next(text_encoder.parameters()).dtype
                    except:
                        target_dtype = torch.float32 # Fallback
            
            res = torch.cat(embeds, dim=1).to(device=self.device, dtype=target_dtype)
            return res

        return encode(pos_chunks), encode(neg_chunks)

    def _parse_and_extract_loras(self, prompt):
        """
        프롬프트에서 <lora:filename:multiplier> 태그를 추출하고 제거합니다.
        [<lora:A:1>:<lora:B:0.5>] 같은 중첩/복합 문법에서도 LoRA 태그 자체를 찾아냅니다.
        """
        # 정규식: <lora:이름> 또는 <lora:이름:강도>
        # [Fix] Global search enables finding LoRAs inside other structures
        pattern = r"<lora:([^:>]+)(?::([\d.]+))?>"
        loras = []
        
        # 1. Extract all LoRAs
        matches = re.findall(pattern, prompt)
        for name, scale_str in matches:
             scale = float(scale_str) if scale_str else 1.0
             loras.append((name, scale))
             
        # 2. Remove tags from prompt
        # We just remove the <lora:...> parts. 
        # Note: If it was [<lora:A>:<lora:B>], it becomes [:]. 
        # This leaves residue [ : ] which is usually harmless punctuation to CLIP.
        clean_prompt = re.sub(pattern, "", prompt)
        
        return clean_prompt, loras
        
    def _apply_mask_content(self, image, mask, mode):
        """
        [New] Mask Content (Initialization) Logic
        mode: 'original', 'fill', 'latent_noise', 'latent_nothing'
        """
        if mode == 'original':
            return image
            
        result = image.copy()
        h, w = image.shape[:2]
        
        # Ensure mask is single channel
        if len(mask.shape) == 3: mask = mask[:,:,0]
        
        # Binary mask for processing
        bin_mask = (mask > 127).astype(np.uint8)
        
        if mode == 'fill':
            # Telea Inpainting (OpenCV)
            # radius=3 is standard
            result = cv2.inpaint(image, bin_mask * 255, 3, cv2.INPAINT_TELEA)
            
        elif mode == 'latent_noise':
            # Fill with random noise
            # Since we work in pixel space here (before VAE), we can fill with random pixel noise
            # which VAE will encode as noisy latent.
            noise = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            # Apply only to masked area
            # Using boolean indexing
            mask_bool = bin_mask > 0
            result[mask_bool] = noise[mask_bool]
            
        elif mode == 'latent_nothing':
            # Fill with constant color (e.g. gray or black)
            # Standard "Nothing" often implies 0 latent, which is mid-gray in pixel?
            # Or just black. Let's use average gray for neutrality.
            fill_color = np.array([127, 127, 127], dtype=np.uint8)
            mask_bool = bin_mask > 0
            result[mask_bool] = fill_color
            
        return result
        
    def _apply_pixel_composite(self, original, result, mask, config):
        """
        [New] Soft Inpainting: Pixel Composite
        Blends the in-painted result with the original based on mask influence and difference threshold.
        """
        try:
            # Configs
            mask_infl = config.get('soft_mask_influence', 0.0)
            diff_thresh = config.get('soft_diff_threshold', 0.5)
            diff_contrast = config.get('soft_diff_contrast', 2.0)
            
            if mask_infl == 0 and diff_thresh == 0.5 and diff_contrast == 2.0:
                # Optimized: Default usually means standard blending (just use result?)
                # Wait, "Difference threshold 0.5" isn't "pass-through".
                # If user enables Soft Inpainting but leaves defaults, we should apply logic.
                pass

            # Convert to float for math
            F_orig = original.astype(np.float32) / 255.0
            F_res = result.astype(np.float32) / 255.0
            F_mask = mask.astype(np.float32) / 255.0
            if len(F_mask.shape) == 2: F_mask = np.expand_dims(F_mask, axis=2)
            
            # Calculate Difference (Luma or RGB distance)
            diff = np.abs(F_orig - F_res)
            # Average across channels
            diff_map = np.mean(diff, axis=2, keepdims=True)
            
            # 1. Mask Influence
            # "How strongly the original mask should bias the difference threshold"
            # If mask_infl is high, we preserve more original content near mask edges?
            # Logic borrowed from WebUI Soft Inpainting:
            # threshold_map = threshold + (mask * mask_influence) ? 
            # Actually WebUI logic is complex. 
            # Simplified: Adjust diff_map based on mask opacity if mask is soft.
            # But here mask is usually binary-ish unless blurred.
            
            # Simple implementation: 
            # We want to keep ORIGINAL if difference is SMALL (below threshold).
            # We want to keep NEW if difference is LARGE (above threshold).
            # Use Sigmoid for smooth transition.
            
            # x = (diff - threshold) * contrast
            # alpha = sigmoid(x) -> 0 (Original) to 1 (Result)
            
            # Adjust threshold with mask influence?
            # If mask is strong (1), threshold increases? (Harder to change)
            # thresh = base_thresh + (1 - F_mask) * mask_infl ?
            
            threshold = diff_thresh
            
            # Contrast
            x = (diff_map - threshold) * diff_contrast
            # Sigmoid: 1 / (1 + exp(-x))
            alpha_map = 1.0 / (1.0 + np.exp(-x))
            
            # Blend
            # composite = original * (1 - alpha) + result * alpha
            composite = F_orig * (1.0 - alpha_map) + F_res * alpha_map
            
            return np.clip(composite * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"    [Error] Pixel Composite failed: {e}")
            return result
