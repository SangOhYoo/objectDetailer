import sys
import os
from core.config import config_instance as cfg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QCheckBox, QTextEdit, QGroupBox, 
                             QDoubleSpinBox, QSlider, QScrollArea, QSpinBox, 
                             QRadioButton, QButtonGroup, QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt

class AdetailerUnitWidget(QWidget):
    def __init__(self, unit_name="패스 1"):
        super().__init__()
        self.unit_name = unit_name
        self.settings = {}  # 슬라이더/스핀박스 위젯 참조 저장
        
        # 저장된 설정 로드 (없으면 빈 딕셔너리)
        self.saved_config = cfg.get('ui_settings', self.unit_name) or {}
        
        self.init_ui()

    def init_ui(self):
        # [New] Scroll is still useful for small screens, even with Tabs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        # [Fix] Re-enable Horizontal Scroll allows narrowing the panel without losing access to controls
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        content_widget = QWidget()
        self.main_layout = QVBoxLayout(content_widget)
        self.main_layout.setSpacing(5) # Compact spacing
        self.main_layout.setContentsMargins(2, 2, 2, 2) # Reduced margins

        # [Fix] Ratio Unit Conversion
        if 'min_face_ratio' in self.saved_config:
             val = self.saved_config['min_face_ratio']
             if val <= 1.0: self.saved_config['min_face_ratio'] = val * 100.0
        if 'max_face_ratio' in self.saved_config:
             val = self.saved_config['max_face_ratio']
             if val <= 1.0: self.saved_config['max_face_ratio'] = val * 100.0

        # ==========================================================
        # 1. TOP AREA (Fixed)
        # ==========================================================
        top_group = QGroupBox("기본 설정 (Basic)")
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(5, 5, 5, 5)

        # Row 1: Enable | Mode | Auto Prompt (Merged for compactness)
        row1_layout = QHBoxLayout()
        
        self.chk_enable = QCheckBox(f"활성화 ({self.unit_name})") # Shortened label
        self.chk_enable.setStyleSheet("font-weight: bold; color: #3498db;")
        self.chk_enable.setChecked(self.saved_config.get('enabled', ("1" in self.unit_name)))
        row1_layout.addWidget(self.chk_enable)
        
        # Mode Radios
        self.radio_yolo = QRadioButton("YOLO")
        self.radio_sam = QRadioButton("SAM3")
        if self.saved_config.get('use_sam', False): self.radio_sam.setChecked(True)
        else: self.radio_yolo.setChecked(True)
        
        bg = QButtonGroup(self)
        bg.addButton(self.radio_yolo)
        bg.addButton(self.radio_sam)
        
        row1_layout.addWidget(self.radio_yolo)
        row1_layout.addWidget(self.radio_sam)
        
        # Auto Prompt (Moved up)
        self.chk_auto_prompt = QCheckBox("✨ 자동 프롬프트")
        self.chk_auto_prompt.setChecked(self.saved_config.get('auto_prompt_injection', True))
        row1_layout.addWidget(self.chk_auto_prompt)
        
        row1_layout.addStretch()
        top_layout.addLayout(row1_layout)

        # Row 2: Model | YOLO Classes (Merged)
        row2_layout = QHBoxLayout()
        
        # Model
        self.combo_model = QComboBox()
        self.combo_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Scan Paths
        search_paths = [cfg.get_path('sam')]
        # [New] Add User External Path
        ext_path = r"D:\AI_Models\adetailer"
        if os.path.exists(ext_path):
            search_paths.append(ext_path)
            
        found_models = set()
        for p in search_paths:
            if p and os.path.exists(p):
                for f in os.listdir(p):
                    if f.endswith('.pt') or f.endswith('.pth'):
                        found_models.add(f)
        
        if found_models:
            self.combo_model.addItems(sorted(list(found_models)))
        else:
            self.combo_model.addItems(["face_yolov8n.pt", "person_yolov8n-seg.pt"])
            
        # Restore Model
        saved_model = self.saved_config.get('detector_model', '')
        if saved_model:
            idx = self.combo_model.findText(saved_model)
            if idx >= 0: self.combo_model.setCurrentIndex(idx)
            
        row2_layout.addWidget(QLabel("모델:"))
        row2_layout.addWidget(self.combo_model, 1) # Stretch 1

        # Classes
        self.txt_yolo_classes = QTextEdit()
        self.txt_yolo_classes.setPlaceholderText("YOLO Classes (e.g. cat)")
        self.txt_yolo_classes.setMaximumHeight(26) # Single line look
        self.txt_yolo_classes.setText(self.saved_config.get('yolo_classes', ""))
        # [Fix] Remove inline style to allow theme border
        self.txt_yolo_classes.setObjectName("yolo_classes")
        
        row2_layout.addWidget(QLabel("클래스:"))
        row2_layout.addWidget(self.txt_yolo_classes, 1) # Stretch 1
        
        top_layout.addLayout(row2_layout)

        # Row 3: Prompts (Generous Size)
        # Positive
        self.txt_pos = QTextEdit()
        self.txt_pos.setPlaceholderText("Positive Prompt (긍정 프롬프트)")
        self.txt_pos.setText(self.saved_config.get('pos_prompt', ""))
        self.txt_pos.setMinimumHeight(60) # Generous height
        self.txt_pos.setMaximumHeight(80)
        # [Fix] Use ObjectName for theming instead of inline style
        self.txt_pos.setObjectName("pos_prompt")
        
        # Negative
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlaceholderText("Negative Prompt (부정 프롬프트)")
        self.txt_neg.setText(self.saved_config.get('neg_prompt', ""))
        self.txt_neg.setMinimumHeight(45) # Slightly smaller than pos but generous
        self.txt_neg.setMaximumHeight(60)
        # [Fix] Use ObjectName for theming instead of inline style
        self.txt_neg.setObjectName("neg_prompt")
        
        top_layout.addWidget(self.txt_pos)
        top_layout.addWidget(self.txt_neg)
        
        top_group.setLayout(top_layout)
        self.main_layout.addWidget(top_group)

        # ==========================================================
        # 2. TAB AREA
        # ==========================================================
        from PyQt6.QtWidgets import QTabWidget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #bdc3c7; }")
        
        # --- TAB 1: Detection & Mask ---
        tab1 = QWidget()
        t1_layout = QVBoxLayout(tab1)
        t1_layout.setContentsMargins(5,5,5,5)

        # Detection Group
        g_det = QGroupBox("감지 설정 (Detection)")
        l_det = QGridLayout()
        # Gender/Edge/Anatomy
        self.combo_gender = QComboBox()
        self.combo_gender.addItems(["All", "Male", "Female"])
        saved_gender = self.saved_config.get('gender_filter', "All")
        self.combo_gender.setCurrentText("All" if saved_gender=="All" else saved_gender)
        
        self.chk_ignore_edge = QCheckBox("Edge무시")
        self.chk_ignore_edge.setChecked(self.saved_config.get('ignore_edge_touching', False))
        self.chk_anatomy = QCheckBox("해부학")
        self.chk_anatomy.setChecked(self.saved_config.get('anatomy_check', True))
        
        # [New] Pose Rotation (Lying Body)
        self.chk_pose_rotation = QCheckBox("Pose회전")
        self.chk_pose_rotation.setToolTip("YOLO Pose를 사용하여 누워있는 신체의 머리 방향을 감지하고 회전합니다.")
        self.chk_pose_rotation.setChecked(self.saved_config.get('use_pose_rotation', False))
        self.chk_pose_rotation.setStyleSheet("color: #8e44ad; font-weight: bold;")

        l_det.addWidget(QLabel("성별:"), 0, 0)
        l_det.addWidget(self.combo_gender, 0, 1, 1, 3) # Span 3 columns
        
        # Sliders
        self.add_slider_row(l_det, 1, "신뢰도:", "conf_thresh", 0.0, 1.0, 0.35, 0.01)
        self.add_slider_row(l_det, 2, "최소(%):", "min_face_ratio", 0.0, 100.0, 1.0, 0.1)
        self.add_slider_row(l_det, 3, "최대(%):", "max_face_ratio", 0.0, 100.0, 100.0, 0.1)
        
        # Sort & Limit
        l_det.addWidget(QLabel("최대 수:"), 4, 0)
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setValue(self.saved_config.get('max_det', 20))
        l_det.addWidget(self.spin_top_k, 4, 1)
        
        # [Unified Bottom Section: Single Row for Sort & Checkboxes]
        from PyQt6.QtWidgets import QFrame
        
        l_bottom_row = QHBoxLayout()
        l_bottom_row.setContentsMargins(0, 10, 0, 0)
        l_bottom_row.setSpacing(10)

        # 1. Sort Controls (Left)
        l_bottom_row.addWidget(QLabel("정렬:"))
        
        self.bg_sort = QButtonGroup(self)
        self.radio_sort_lr = QRadioButton("좌→우"); self.bg_sort.addButton(self.radio_sort_lr)
        self.radio_sort_center = QRadioButton("중앙"); self.bg_sort.addButton(self.radio_sort_center)
        self.radio_sort_area = QRadioButton("크기"); self.bg_sort.addButton(self.radio_sort_area)
        self.radio_sort_tb = QRadioButton("위→아래"); self.bg_sort.addButton(self.radio_sort_tb) 
        self.radio_sort_conf = QRadioButton("신뢰도"); self.bg_sort.addButton(self.radio_sort_conf)
        
        saved_sort = self.saved_config.get('sort_method', '신뢰도')
        if '좌' in saved_sort: self.radio_sort_lr.setChecked(True)
        elif '중앙' in saved_sort: self.radio_sort_center.setChecked(True)
        elif '영역' in saved_sort: self.radio_sort_area.setChecked(True)
        elif '위' in saved_sort: self.radio_sort_tb.setChecked(True)
        else: self.radio_sort_conf.setChecked(True)
        
        radio_style = "QRadioButton { font-size: 12px; padding: 2px; }"
        for r in [self.radio_sort_lr, self.radio_sort_center, self.radio_sort_area, self.radio_sort_tb, self.radio_sort_conf]:
            r.setStyleSheet(radio_style)
            l_bottom_row.addWidget(r)
            
        # 2. Vertical Separator
        v_line = QFrame()
        v_line.setFrameShape(QFrame.Shape.VLine)
        v_line.setFrameShadow(QFrame.Shadow.Sunken)
        l_bottom_row.addWidget(v_line)
        
        # 3. Checkboxes (Right)
        chk_style = "QCheckBox { font-size: 13px; padding: 5px; }"
        self.chk_ignore_edge.setStyleSheet(chk_style)
        self.chk_anatomy.setStyleSheet(chk_style)
        self.chk_pose_rotation.setStyleSheet(chk_style + " color: #8e44ad; font-weight: bold;")
        
        l_bottom_row.addWidget(self.chk_ignore_edge)
        l_bottom_row.addWidget(self.chk_anatomy)
        l_bottom_row.addWidget(self.chk_pose_rotation)
        
        l_bottom_row.addStretch()

        # Add Combined Single Row to Grid (Row 5, Span 4)
        l_det.addLayout(l_bottom_row, 5, 0, 1, 4)
        g_det.setLayout(l_det)
        t1_layout.addWidget(g_det)
        
        # Mask Group
        g_mask = QGroupBox("마스크 전처리")
        l_mask = QGridLayout()
        self.add_slider_row(l_mask, 0, "확장(Dil):", "mask_dilation", -64, 64, 4, 1)
        self.add_slider_row(l_mask, 1, "침식(Ero):", "mask_erosion", 0, 64, 0, 1)
        self.add_slider_row(l_mask, 2, "블러(Blu):", "mask_blur", 0, 64, 12, 1)
        self.add_slider_row(l_mask, 3, "X 오프셋:", "x_offset", -100, 100, 0, 1)
        self.add_slider_row(l_mask, 4, "Y 오프셋:", "y_offset", -100, 100, 0, 1)
        g_mask.setLayout(l_mask)
        t1_layout.addWidget(g_mask)
        t1_layout.addStretch()
        
        # SAM Settings (Optional)
        g_sam = QGroupBox("SAM 설정")
        l_sam = QGridLayout()
        self.add_slider_row(l_sam, 0, "Points:", "sam_points_per_side", 1, 64, 32, 1)
        self.add_slider_row(l_sam, 1, "IOU:", "sam_pred_iou_thresh", 0.0, 1.0, 0.88, 0.01)
        self.add_slider_row(l_sam, 2, "Stability:", "sam_stability_score_thresh", 0.0, 1.0, 0.95, 0.01)
        g_sam.setLayout(l_sam)
        t1_layout.addWidget(g_sam)
        
        self.tabs.addTab(tab1, "감지 (Detect)")

        # --- TAB 2: Inpaint & ControlNet ---
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        
        # Inpaint Group
        g_inp = QGroupBox("인페인팅 설정")
        l_inp = QGridLayout()
        self.add_slider_row(l_inp, 0, "디노이징:", "denoising_strength", 0.0, 1.0, 0.4, 0.01)
        self.add_slider_row(l_inp, 1, "문맥확장:", "context_expand_factor", 1.0, 3.0, 1.0, 0.1)
        self.add_slider_row(l_inp, 2, "패딩(px):", "crop_padding", 0, 256, 32, 1)
        
        # Resolution
        l_res = QHBoxLayout()
        l_res.addWidget(QLabel("해상도:"))
        self.spin_inpaint_w = QSpinBox(); self.spin_inpaint_w.setRange(0, 2048); 
        self.spin_inpaint_w.setValue(self.saved_config.get('inpaint_width', 0))
        self.spin_inpaint_h = QSpinBox(); self.spin_inpaint_h.setRange(0, 2048);
        self.spin_inpaint_h.setValue(self.saved_config.get('inpaint_height', 0))
        l_res.addWidget(self.spin_inpaint_w); l_res.addWidget(QLabel("x")); l_res.addWidget(self.spin_inpaint_h)
        l_inp.addLayout(l_res, 3, 0, 1, 3)
        
        # Mask Merge & Special
        self.combo_mask_merge = QComboBox(); self.combo_mask_merge.addItems(["None", "Merge", "Merge and Invert"])
        # Legacy bool support
        saved_merge = self.saved_config.get('mask_merge_mode', "None")
        if isinstance(saved_merge, bool): saved_merge = "Merge" if saved_merge else "None"
        self.combo_mask_merge.setCurrentText(saved_merge)
        
        l_inp.addWidget(QLabel("병합:"), 4, 0)
        l_inp.addWidget(self.combo_mask_merge, 4, 1)
        
        self.chk_noise_mask = QCheckBox("노이즈 마스크")
        self.chk_noise_mask.setChecked(self.saved_config.get('use_noise_mask', False))
        l_inp.addWidget(self.chk_noise_mask, 5, 0)
        
        self.chk_auto_rotate = QCheckBox("자동 회전")
        self.chk_auto_rotate.setChecked(self.saved_config.get('auto_rotate', True))
        l_inp.addWidget(self.chk_auto_rotate, 5, 1)
        
        self.combo_color_fix = QComboBox(); self.combo_color_fix.addItems(["None", "Wavelet", "Adain"])
        self.combo_color_fix.setCurrentText(self.saved_config.get('color_fix', "None"))
        l_inp.addWidget(QLabel("색감보정:"), 6, 0)
        l_inp.addWidget(self.combo_color_fix, 6, 1)
        
        g_inp.setLayout(l_inp)
        t2_layout.addWidget(g_inp)
        
        # ControlNet Group
        g_cn = QGroupBox("ControlNet")
        l_cn = QGridLayout()
        
        self.combo_cn_model = QComboBox(); self.combo_cn_model.addItem("None")
        cn_dir = cfg.get_path('controlnet')
        if cn_dir and os.path.exists(cn_dir):
            try:
                cn_models = [f for f in os.listdir(cn_dir) if f.endswith(('.pth', '.safetensors', '.bin'))]
                self.combo_cn_model.addItems(sorted(cn_models))
            except: pass
        self.combo_cn_model.setCurrentText(self.saved_config.get('control_model', "None"))
        
        # [Fix] control_module 복구
        self.combo_cn_module = QComboBox() 
        self.combo_cn_module.addItems(["None", "openpose", "canny", "depth_midas", "openpose_full", "softedge_pidinet", "scribble_hed"])
        self.combo_cn_module.setCurrentText(self.saved_config.get('control_module', "None"))

        l_cn.addWidget(QLabel("모델:"), 0, 0)
        l_cn.addWidget(self.combo_cn_model, 0, 1)
        l_cn.addWidget(QLabel("전처리:"), 1, 0)
        l_cn.addWidget(self.combo_cn_module, 1, 1)
        
        self.add_slider_row(l_cn, 2, "가중치:", "control_weight", 0.0, 2.0, 1.0, 0.1)
        self.add_slider_row(l_cn, 3, "시작:", "guidance_start", 0.0, 1.0, 0.0, 0.05)
        self.add_slider_row(l_cn, 4, "종료:", "guidance_end", 0.0, 1.0, 1.0, 0.05)
        
        g_cn.setLayout(l_cn)
        t2_layout.addWidget(g_cn)
        t2_layout.addStretch()
        self.tabs.addTab(tab2, "인페인팅 (Inpaint)")
        
        # --- TAB 3: BMAB & Composition ---
        tab3 = QWidget()
        t3_layout = QVBoxLayout(tab3)
        
        # BMAB Preprocess
        g_bmab = QGroupBox("BMAB 이미지 보정")
        l_bmab = QGridLayout()
        self.add_slider_row(l_bmab, 0, "대비:", "bmab_contrast", 0.0, 3.0, 1.0, 0.05)
        self.add_slider_row(l_bmab, 1, "밝기:", "bmab_brightness", 0.0, 3.0, 1.0, 0.05)
        self.add_slider_row(l_bmab, 2, "선명도:", "bmab_sharpness", 0.0, 3.0, 1.0, 0.05)
        self.add_slider_row(l_bmab, 3, "색온도:", "bmab_color_temp", -100.0, 100.0, 0.0, 1.0)
        self.add_slider_row(l_bmab, 4, "노이즈:", "bmab_noise_alpha", 0.0, 1.0, 0.0, 0.01)
        self.add_slider_row(l_bmab, 5, "엣지(강도):", "bmab_edge_strength", 0.0, 1.0, 0.0, 0.05)
        self.add_slider_row(l_bmab, 6, "엣지(Low):", "bmab_edge_low", 0, 255, 50, 1)
        self.add_slider_row(l_bmab, 7, "엣지(High):", "bmab_edge_high", 0, 255, 200, 1)
        g_bmab.setLayout(l_bmab)
        t3_layout.addWidget(g_bmab)
        
        # Composition
        g_comp = QGroupBox("캔버스 확장 (Resize by Person)")
        l_comp = QGridLayout()
        self.chk_resize_enable = QCheckBox("활성화")
        self.chk_resize_enable.setChecked(self.saved_config.get('resize_enable', False))
        l_comp.addWidget(self.chk_resize_enable, 0, 0)
        
        self.add_slider_row(l_comp, 1, "목표 비율:", "resize_ratio", 0.1, 1.0, 0.6, 0.05)
        
        self.combo_resize_align = QComboBox(); self.combo_resize_align.addItems(["Center", "Bottom", "Top"])
        self.combo_resize_align.setCurrentText(self.saved_config.get('resize_align', "Center"))
        l_comp.addWidget(QLabel("정렬:"), 2, 0)
        l_comp.addWidget(self.combo_resize_align, 2, 1)
        g_comp.setLayout(l_comp)
        t3_layout.addWidget(g_comp)
        t3_layout.addStretch()
        self.tabs.addTab(tab3, "BMAB (Effect)")

        # --- TAB 4: Advanced ---
        tab4 = QWidget()
        t4_layout = QVBoxLayout(tab4)
        
        g_adv = QGroupBox("고급 재정의 (Overrides)")
        l_adv = QGridLayout()
        
        # CKPT/VAE
        self.chk_sep_ckpt = QCheckBox("CKPT"); self.chk_sep_ckpt.setChecked(self.saved_config.get('sep_ckpt', False))
        self.combo_sep_ckpt = QComboBox(); self.combo_sep_ckpt.addItem("Use Global")
        ckpt_dir = cfg.get_path('checkpoint')
        if ckpt_dir: self.combo_sep_ckpt.addItems([f for f in os.listdir(ckpt_dir) if f.endswith(('.ckpt', '.safetensors'))])
        self.combo_sep_ckpt.setCurrentText(self.saved_config.get('sep_ckpt_name', 'Use Global'))
        
        self.chk_sep_vae = QCheckBox("VAE"); self.chk_sep_vae.setChecked(self.saved_config.get('sep_vae', False))
        self.combo_sep_vae = QComboBox(); self.combo_sep_vae.addItem("Use Global")
        vae_dir = cfg.get_path('vae')
        if vae_dir: self.combo_sep_vae.addItems([f for f in os.listdir(vae_dir) if f.endswith(('.pt','.ckpt','.safetensors'))])
        self.combo_sep_vae.setCurrentText(self.saved_config.get('sep_vae_name', 'Use Global'))
        
        l_adv.addWidget(self.chk_sep_ckpt, 0, 0); l_adv.addWidget(self.combo_sep_ckpt, 0, 1)
        l_adv.addWidget(self.chk_sep_vae, 0, 2); l_adv.addWidget(self.combo_sep_vae, 0, 3)
        
        # Sampler
        self.chk_sep_sampler = QCheckBox("Sampler"); self.chk_sep_sampler.setChecked(self.saved_config.get('sep_sampler', False))
        self.combo_sep_sampler = QComboBox(); self.combo_sep_sampler.addItems(["Euler a", "DPM++ 2M", "DPM++ SDE", "DDIM"])
        self.combo_sep_scheduler = QComboBox(); self.combo_sep_scheduler.addItems(["Karras", "Exponential", "Automatic"])
        
        # Sampler Restore
        saved_sampler_full = self.saved_config.get('sampler_name', "Euler a Automatic")
        schedulers = ["Karras", "Exponential", "Automatic"]
        found_sch = "Automatic"; found_sam = saved_sampler_full
        for s in schedulers:
            if saved_sampler_full.endswith(s): found_sch = s; found_sam = saved_sampler_full.replace(s, "").strip(); break
        self.combo_sep_sampler.setCurrentText(found_sam)
        self.combo_sep_scheduler.setCurrentText(found_sch)
        
        l_adv.addWidget(self.chk_sep_sampler, 1, 0)
        l_adv.addWidget(self.combo_sep_sampler, 1, 1)
        l_adv.addWidget(self.combo_sep_scheduler, 1, 2)
        
        # Steps/CFG/Clip
        l_sub = QHBoxLayout()
        self.chk_sep_steps=QCheckBox("Steps"); self.chk_sep_steps.setChecked(self.saved_config.get('sep_steps', False))
        self.spin_sep_steps=QSpinBox(); self.spin_sep_steps.setValue(self.saved_config.get('steps', 20))
        self.chk_sep_cfg=QCheckBox("CFG"); self.chk_sep_cfg.setChecked(self.saved_config.get('sep_cfg', False))
        self.spin_sep_cfg=QDoubleSpinBox(); self.spin_sep_cfg.setValue(self.saved_config.get('cfg_scale', 7.0))
        self.chk_sep_clip=QCheckBox("Clip"); self.chk_sep_clip.setChecked(self.saved_config.get('sep_clip', False))
        self.spin_clip=QSpinBox(); self.spin_clip.setRange(1,12); self.spin_clip.setValue(self.saved_config.get('clip_skip', 2))
        
        l_sub.addWidget(self.chk_sep_steps); l_sub.addWidget(self.spin_sep_steps)
        l_sub.addWidget(self.chk_sep_cfg); l_sub.addWidget(self.spin_sep_cfg)
        l_sub.addWidget(self.chk_sep_clip); l_sub.addWidget(self.spin_clip)
        l_adv.addLayout(l_sub, 2, 0, 1, 4)
        
        self.chk_hires = QCheckBox("Hires Fix"); self.chk_hires.setChecked(self.saved_config.get('use_hires_fix', False))
        self.chk_sep_noise = QCheckBox("Sep Noise"); self.chk_sep_noise.setChecked(self.saved_config.get('sep_noise', False))
        self.chk_restore_face = QCheckBox("Restore Face"); self.chk_restore_face.setChecked(self.saved_config.get('restore_face', False))
        l_adv.addWidget(self.chk_hires, 3, 0)
        l_adv.addWidget(self.chk_sep_noise, 3, 1)
        l_adv.addWidget(self.chk_restore_face, 3, 2, 1, 2)
        
        g_adv.setLayout(l_adv)
        t4_layout.addWidget(g_adv)
        t4_layout.addStretch()
        self.tabs.addTab(tab4, "고급 (Adv)")

        # Finish Setup
        self.main_layout.addWidget(self.tabs)
        scroll.setWidget(content_widget)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        outer.addWidget(scroll)
        
        # Signals
        self.combo_sep_ckpt.currentTextChanged.connect(self.on_local_ckpt_changed)
        self.chk_sep_ckpt.toggled.connect(self.on_sep_ckpt_toggled)

    def add_slider_row(self, layout, row, label_text, key, min_val, max_val, default_val, step, start_col=0):
        label = QLabel(label_text)
        slider = QSlider(Qt.Orientation.Horizontal)
        
        loaded_val = self.saved_config.get(key, default_val)
        
        is_float = isinstance(default_val, float)
        scale = 100 if is_float else 1
        
        slider.setRange(int(min_val * scale), int(max_val * scale))
        
        if is_float:
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
        else:
            spin = QSpinBox()
            
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        # [Fix] Increased width to prevent text cutoff (was 60)
        spin.setFixedWidth(80)
        
        spin.setValue(loaded_val)
        slider.setValue(int(loaded_val * scale))
        
        # [Fix] 슬라이더 -> 스핀박스 (타입 에러 방지)
        if is_float:
            slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        else:
            slider.valueChanged.connect(lambda v: spin.setValue(int(v / scale)))
            
        # 스핀박스 -> 슬라이더 (값 변경 시 즉시 반영)
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * scale)))
        
        layout.addWidget(label, row, start_col)
        layout.addWidget(slider, row, start_col + 1)
        layout.addWidget(spin, row, start_col + 2)
        
        self.settings[key] = spin

    def on_local_ckpt_changed(self, text):
        """개별 체크포인트 변경 시 UI 업데이트"""
        if self.chk_sep_ckpt.isChecked() and text != "Use Global":
            self.apply_model_presets(text)

    def on_sep_ckpt_toggled(self, checked):
        """개별 체크포인트 사용 여부 토글 시 UI 업데이트"""
        if checked:
            self.on_local_ckpt_changed(self.combo_sep_ckpt.currentText())
        # 체크 해제 시 글로벌 설정은 MainWindow에서 다시 전파되거나 다음 글로벌 변경 시 적용됨

    def on_global_model_changed(self, text):
        """글로벌 모델 변경 시 (개별 설정이 꺼져있거나 Use Global일 때) UI 업데이트"""
        if not self.chk_sep_ckpt.isChecked() or self.combo_sep_ckpt.currentText() == "Use Global":
            self.apply_model_presets(text)

    def apply_model_presets(self, model_name):
        """모델 이름에 따라 해상도 등 프리셋 자동 적용 (SDXL vs SD1.5)"""
        name = model_name.lower()
        is_sdxl = "xl" in name or "pony" in name
        
        if is_sdxl:
            # SDXL Defaults: 1024x1024
            if self.spin_inpaint_w.value() == 512 or self.spin_inpaint_w.value() == 0: self.spin_inpaint_w.setValue(1024)
            if self.spin_inpaint_h.value() == 512 or self.spin_inpaint_h.value() == 0: self.spin_inpaint_h.setValue(1024)
        else:
            # SD1.5 Defaults: 512x512
            if self.spin_inpaint_w.value() == 1024: self.spin_inpaint_w.setValue(512)
            if self.spin_inpaint_h.value() == 1024: self.spin_inpaint_h.setValue(512)

    def get_config(self):
        """Configs.py Key 완전 일치 및 누락 기능 복구 완료"""
        cfg = {
            'enabled': self.chk_enable.isChecked(),
            'detector_model': self.combo_model.currentText(),
            'yolo_classes': self.txt_yolo_classes.toPlainText(),
            'use_sam': self.radio_sam.isChecked(),
            
            'auto_prompt_injection': self.chk_auto_prompt.isChecked(),
            'gender_filter': self.combo_gender.currentText().split()[0],
            'ignore_edge_touching': self.chk_ignore_edge.isChecked(),
            'anatomy_check': self.chk_anatomy.isChecked(),
            'use_pose_rotation': self.chk_pose_rotation.isChecked(), # [New]
            'auto_rotate': self.chk_auto_rotate.isChecked(),
            'color_fix': self.combo_color_fix.currentText(),
            'use_hires_fix': self.chk_hires.isChecked(),
            
            'pos_prompt': self.txt_pos.toPlainText(),
            'neg_prompt': self.txt_neg.toPlainText(),
            'max_det': self.spin_top_k.value(),
            
            'sort_method': '위치(좌에서 우)' if self.radio_sort_lr.isChecked() else \
                           '위치 (중앙에서 바깥)' if self.radio_sort_center.isChecked() else \
                           '영역 (대형에서 소형)' if self.radio_sort_area.isChecked() else \
                           '위치(위에서 아래)' if self.radio_sort_tb.isChecked() else '신뢰도',
            
            'use_controlnet': self.combo_cn_model.currentText() != "None",
            'control_model': self.combo_cn_model.currentText(),
            'control_module': self.combo_cn_module.currentText(), # [복구됨]
            'sep_noise': self.chk_sep_noise.isChecked(),
            
            'inpaint_width': self.spin_inpaint_w.value(),
            'inpaint_height': self.spin_inpaint_h.value(),
            'use_noise_mask': self.chk_noise_mask.isChecked(),
            'inpaint_width': self.spin_inpaint_w.value(),
            'inpaint_height': self.spin_inpaint_h.value(),
            'use_noise_mask': self.chk_noise_mask.isChecked(),
            # Sliders are handled below but let's be explicit if needed or trust the slider loop
            # Sliders loop covers 'context_expand_factor' because it was added via add_slider_row
            # Sliders loop covers 'context_expand_factor' because it was added via add_slider_row
            'mask_merge_mode': self.combo_mask_merge.currentText(), # [Updated] String value

            # [New] Composition
            'resize_enable': self.chk_resize_enable.isChecked(),
            'resize_align': self.combo_resize_align.currentText(),

            # --- 고급 오버라이드 ---
            'sep_ckpt': self.chk_sep_ckpt.isChecked(),
            'sep_ckpt_name': self.combo_sep_ckpt.currentText(),
            'sep_vae': self.chk_sep_vae.isChecked(),
            'sep_vae_name': self.combo_sep_vae.currentText(),
            
            'sep_sampler': self.chk_sep_sampler.isChecked(),
            'sampler_name': f"{self.combo_sep_sampler.currentText()} {self.combo_sep_scheduler.currentText()}", 
            
            'sep_steps': self.chk_sep_steps.isChecked(),
            'steps': self.spin_sep_steps.value(),
            
            'sep_cfg': self.chk_sep_cfg.isChecked(),
            'cfg_scale': self.spin_sep_cfg.value(),
            
            'sep_clip': self.chk_sep_clip.isChecked(),
            'clip_skip': self.spin_clip.value(),
            
            'restore_face': self.chk_restore_face.isChecked(),
        }

        # 슬라이더 값들 병합 (Max Face Ratio 등 포함)
        for key, widget in self.settings.items():
            val = widget.value()
            # [Fix] % 단위 UI 값을 Ratio(0~1)로 변환하여 로직에 전달
            if key in ['min_face_ratio', 'max_face_ratio']:
                val /= 100.0
            cfg[key] = val
            
        cfg['seed'] = -1
        return cfg