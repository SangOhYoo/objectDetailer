
import sys
import os
from core.config import config_instance as cfg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QCheckBox, QTextEdit, QGroupBox,
                             QDoubleSpinBox, QSlider, QScrollArea, QSpinBox,
                             QRadioButton, QButtonGroup, QGridLayout, QSizePolicy, QPushButton,
                             QAbstractSpinBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from core.detail_daemon import make_schedule
from ui.graph_widget import ScheduleGraphWidget
from ui.styles import ModernTheme

class AdetailerUnitWidget(QWidget):
    # [Signal] Request Main Window to run detection preview
    preview_requested = pyqtSignal(dict)

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
        self.chk_enable.setObjectName("important_chk")
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
        
        # [New] Reset Button
        self.btn_reset = QPushButton("초기화 (Reset)")
        self.btn_reset.setToolTip("이 패스의 모든 설정을 기본값으로 되돌립니다.")
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        self.btn_reset.setObjectName("warning_btn")
        row1_layout.addWidget(self.btn_reset)

        # [New] Detect Preview Button
        self.btn_detect_preview = QPushButton("탐지 (Detect)")
        self.btn_detect_preview.setToolTip("선택된 이미지에 대해 탐지 테스트만 수행합니다. (인페인팅 건너뜀)")
        self.btn_detect_preview.clicked.connect(self.on_detect_preview_clicked)
        row1_layout.addWidget(self.btn_detect_preview)
        
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
        # [Fix] Use 'files.sam_file' as fallback default if no unit-specific model is saved
        default_model = cfg.get('files', 'sam_file') or "face_yolov8n.pt"
        saved_model = self.saved_config.get('detector_model', default_model)
        
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
        # [Ref] Unlock Max Height for Vertical Expansion
        self.txt_pos.setMinimumHeight(60) 
        self.txt_pos.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.txt_pos.setObjectName("pos_prompt")
        
        # Negative
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlaceholderText("Negative Prompt (부정 프롬프트)")
        self.txt_neg.setText(self.saved_config.get('neg_prompt', ""))
        # [Ref] Unlock Max Height for Vertical Expansion
        self.txt_neg.setMinimumHeight(50)
        self.txt_neg.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.txt_neg.setObjectName("neg_prompt")
        
        # [Fix] Add Stretch factors (Pos: 2, Neg: 1)
        top_layout.addWidget(self.txt_pos, 2)
        top_layout.addWidget(self.txt_neg, 1)
        
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
        t1_layout.setContentsMargins(2, 2, 2, 2)
        t1_layout.setSpacing(4) # Tighter spacing

        # Detection Group
        g_det = QGroupBox("감지 설정 (Detection)")
        l_det = QGridLayout()
        l_det.setContentsMargins(5, 10, 5, 5) # Top margin for title
        l_det.setVerticalSpacing(4) # Tighter vertical gap
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
        self.chk_pose_rotation.setObjectName("purple_chk")

        l_det.addWidget(QLabel("성별:"), 0, 0)
        l_det.addWidget(self.combo_gender, 0, 1, 1, 3) # Span 3 columns
        
        # Sliders
        # [Fix] Use global defaults
        def_conf = cfg.get('defaults', 'conf_thresh') or 0.35
        def_min_face = (cfg.get('defaults', 'min_face_ratio') or 0.01) * 100.0
        def_max_face = (cfg.get('defaults', 'max_face_ratio') or 1.0) * 100.0
        
        # [Ref] Revert to 1-Column Layout (User Request)
        # Row 1: Confidence
        self.add_slider_row(l_det, 1, "신뢰도:", "conf_thresh", 0.0, 1.0, def_conf, 0.01)
        # Row 2: Min Area
        self.add_slider_row(l_det, 2, "최소(%):", "min_face_ratio", 0.0, 100.0, def_min_face, 0.1)
        # Row 3: Max Area
        self.add_slider_row(l_det, 3, "최대(%):", "max_face_ratio", 0.0, 100.0, def_max_face, 0.1)
        
        # [Ref] Max Count Slider UI (Aligned)
        l_det.addWidget(QLabel("최대 수:"), 4, 0)
        
        self.slider_max_det = QSlider(Qt.Orientation.Horizontal)
        self.slider_max_det.setRange(1, 100)
        
        self.spin_top_k = QSpinBox() # Preserve name for compatibility
        self.spin_top_k.setRange(1, 100)
        # [Ref] Match SpinBox width to others (70)
        self.spin_top_k.setFixedWidth(70)
        # [Style] Clean Look
        self.spin_top_k.setObjectName("clean_spin")
        self.spin_top_k.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_top_k.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons) # Explicitly remove buttons too 
        
        # Init value
        val_max_det = self.saved_config.get('max_det', cfg.get('defaults', 'max_det') or 20)
        self.slider_max_det.setValue(int(val_max_det))
        self.spin_top_k.setValue(int(val_max_det))
        
        # Sync
        self.slider_max_det.valueChanged.connect(self.spin_top_k.setValue)
        self.spin_top_k.valueChanged.connect(self.slider_max_det.setValue)
        
        # Add to Grid directly (Col 1 for Slider, Col 2 for SpinBox)
        l_det.addWidget(self.slider_max_det, 4, 1)
        l_det.addWidget(self.spin_top_k, 4, 2)
        
        # [Fix] Stretch slider column to fill 800-1000px width
        l_det.setColumnStretch(1, 1)
        
        # [Unified Bottom Section: Single Row for Sort & Checkboxes]
        from PyQt6.QtWidgets import QFrame
        
        # [Ref] Restore Initialization Logic
        self.bg_sort = QButtonGroup(self)
        self.radio_sort_lr = QRadioButton("좌→우"); self.bg_sort.addButton(self.radio_sort_lr)
        self.radio_sort_rl = QRadioButton("우→좌"); self.bg_sort.addButton(self.radio_sort_rl)
        self.radio_sort_center = QRadioButton("중앙"); self.bg_sort.addButton(self.radio_sort_center)
        self.radio_sort_area = QRadioButton("크기"); self.bg_sort.addButton(self.radio_sort_area)
        self.radio_sort_tb = QRadioButton("위→아래"); self.bg_sort.addButton(self.radio_sort_tb) 
        self.radio_sort_bt = QRadioButton("아래→위"); self.bg_sort.addButton(self.radio_sort_bt)
        self.radio_sort_conf = QRadioButton("신뢰도"); self.bg_sort.addButton(self.radio_sort_conf)

        # [Ref] Restore Initialization Logic (Check config)
        saved_sort = self.saved_config.get('sort_method', cfg.get('defaults', 'sort_method') or '신뢰도')
        if '좌에서 우' in saved_sort: self.radio_sort_lr.setChecked(True)
        elif '우에서 좌' in saved_sort: self.radio_sort_rl.setChecked(True)
        elif '중앙' in saved_sort: self.radio_sort_center.setChecked(True)
        elif '영역' in saved_sort: self.radio_sort_area.setChecked(True)
        elif '아래에서 위' in saved_sort: self.radio_sort_bt.setChecked(True)
        elif '위에서 아래' in saved_sort: self.radio_sort_tb.setChecked(True)
        else: self.radio_sort_conf.setChecked(True)

        # [Ref] Removed manual loop for StyleSheet - handled by Global ModernTheme
        # Properties/IDs set if needed, but defaults are fine.

        # [Ref] Unified Sort & Filter Group (Single Row Compact)
        g_sort_filter = QGroupBox("정렬/필터")
        l_sf = QHBoxLayout()
        l_sf.setContentsMargins(5, 5, 5, 2) # Top margin for title space
        l_sf.setSpacing(6)
        
        # Sort Group 1: Horizontal
        l_sf.addWidget(self.radio_sort_lr)
        l_sf.addWidget(self.radio_sort_rl)
        l_sf.addWidget(self.radio_sort_center)
        
        l_sf.addWidget(QLabel("|")) # Separator
        
        # Sort Group 2: Vertical
        l_sf.addWidget(self.radio_sort_tb)
        l_sf.addWidget(self.radio_sort_bt)
        
        l_sf.addWidget(QLabel("|")) # Separator
        
        # Sort Group 3: Misc
        l_sf.addWidget(self.radio_sort_area)
        l_sf.addWidget(self.radio_sort_conf)
        
        l_sf.addWidget(QLabel("|")) # Separator
        
        # Filters
        l_sf.addWidget(self.chk_ignore_edge)
        l_sf.addWidget(self.chk_anatomy)
        l_sf.addWidget(self.chk_pose_rotation)
        
        l_sf.addStretch() # Fill remaining space

        # [Fix] Set layout for the group box
        g_sort_filter.setLayout(l_sf)

        # Add to Main Grid
        l_det.addWidget(g_sort_filter, 5, 0, 1, 4)
        g_det.setLayout(l_det)
        t1_layout.addWidget(g_det)
        
        # [Moved to dedicated tab]
        
        # ---------------------------------------------------------
        
        # Mask Group
        g_mask = QGroupBox("마스크 전처리")
        l_mask = QGridLayout()
        # [Ref] Revert to 1-Column for Mask
        # Row 0: Dilation
        self.add_slider_row(l_mask, 0, "확장(Dil):", "mask_dilation", -64, 64, 4, 1)
        self.add_slider_row(l_mask, 1, "침식(Ero):", "mask_erosion", 0, 64, 0, 1)
        self.add_slider_row(l_mask, 2, "블러(Blu):", "mask_blur", 0, 64, 12, 1)
        self.add_slider_row(l_mask, 3, "X 오프셋:", "x_offset", -100, 100, 0, 1)
        self.add_slider_row(l_mask, 4, "Y 오프셋:", "y_offset", -100, 100, 0, 1)
        
        l_mask.setVerticalSpacing(2)
        l_mask.setColumnStretch(1, 1) # [Fix]
        
        g_mask.setLayout(l_mask)
        t1_layout.addWidget(g_mask)
        
        # [Ref] Move SAM Settings immediately after Mask (User Request)
        # SAM Settings
        g_sam = QGroupBox("SAM 설정")
        l_sam = QGridLayout()
        # [Ref] Tighter Vertical Spacing and 1-Column Grid
        l_sam.setVerticalSpacing(2)
        self.add_slider_row(l_sam, 0, "Points:", "sam_points_per_side", 1, 64, 32, 1)
        self.add_slider_row(l_sam, 1, "IOU:", "sam_pred_iou_thresh", 0.0, 1.0, 0.88, 0.01)
        self.add_slider_row(l_sam, 2, "Stability:", "sam_stability_score_thresh", 0.0, 1.0, 0.95, 0.01)
        l_sam.setColumnStretch(1, 1) # [Fix]
        g_sam.setLayout(l_sam)
        t1_layout.addWidget(g_sam)

        # Stretch at the end
        t1_layout.addStretch()
        
        self.tabs.addTab(tab1, "감지 (Detect)")

        # --- TAB 2: Inpaint & ControlNet ---
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2) # Vertical Layout for Narrow Panel
        
        # Function to ensure tight layout
        def make_tight(layout):
            layout.setContentsMargins(0,0,0,0)
            layout.setSpacing(4)
            
        # Left Column: Inpaint & Mask Settings
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        t2_layout.setContentsMargins(5,5,5,5)
        
        # [Ref] Flattened Layout (No Left/Right Split)
        
        # Group: Inpaint & Mask
        g_inp = QGroupBox("인페인팅 & 마스크 설정")
        l_inp = QGridLayout()
        
        # 1. Mask Content (Row 0)
        l_inp.addWidget(QLabel("마스크 내용:"), 0, 0)
        l_mask_content = QHBoxLayout()
        make_tight(l_mask_content)
        
        self.bg_mask_content = QButtonGroup(self)
        self.radio_content_fill = QRadioButton("채우기")
        self.radio_content_orig = QRadioButton("원본")
        self.radio_content_noise = QRadioButton("노이즈")
        self.radio_content_nothing = QRadioButton("공백")
        
        # Load Config
        saved_content_str = self.saved_config.get('mask_content', 'original')
        if saved_content_str == 'fill': self.radio_content_fill.setChecked(True)
        elif saved_content_str == 'original': self.radio_content_orig.setChecked(True)
        elif saved_content_str == 'latent_noise': self.radio_content_noise.setChecked(True)
        elif saved_content_str == 'latent_nothing': self.radio_content_nothing.setChecked(True)
        else: self.radio_content_orig.setChecked(True)
        
        self.bg_mask_content.addButton(self.radio_content_fill)
        self.bg_mask_content.addButton(self.radio_content_orig)
        self.bg_mask_content.addButton(self.radio_content_noise)
        self.bg_mask_content.addButton(self.radio_content_nothing)
        
        l_mask_content.addWidget(self.radio_content_fill)
        l_mask_content.addWidget(self.radio_content_orig)
        l_mask_content.addWidget(self.radio_content_noise)
        l_mask_content.addWidget(self.radio_content_nothing)
        l_mask_content.addStretch()
        l_inp.addLayout(l_mask_content, 0, 1, 1, 3) 

        # 2. Inpaint Area (Row 1)
        l_inp.addWidget(QLabel("인페인팅 영역:"), 1, 0)
        l_area = QHBoxLayout()
        make_tight(l_area)
        self.bg_inpaint_area = QButtonGroup(self)
        self.radio_area_whole = QRadioButton("전체 (Whole)")
        self.radio_area_masked = QRadioButton("마스크만 (Masked)")
        
        if self.saved_config.get('inpaint_full_res', False):
             self.radio_area_whole.setChecked(True)
        else:
             self.radio_area_masked.setChecked(True)
             
        self.bg_inpaint_area.addButton(self.radio_area_whole)
        self.bg_inpaint_area.addButton(self.radio_area_masked)
        
        l_area.addWidget(self.radio_area_whole)
        l_area.addWidget(self.radio_area_masked)
        l_area.addStretch()
        l_inp.addLayout(l_area, 1, 1, 1, 3)
        
        # 3. Sliders (Row 2~)
        # [Fix] Use global defaults
        def_denoise = cfg.get('defaults', 'denoise') or 0.4
        def_ctx = cfg.get('defaults', 'padding') or 1.0 
        
        self.add_slider_row(l_inp, 2, "Denoise:", "denoising_strength", 0.0, 1.0, def_denoise, 0.01)
        self.add_slider_row(l_inp, 3, "Context:", "context_expand_factor", 1.0, 3.0, def_ctx, 0.1)
        self.add_slider_row(l_inp, 4, "Padding:", "crop_padding", 0, 256, 32, 1)
        
        l_inp.setVerticalSpacing(2)
        l_inp.setColumnStretch(1, 1) # [Fix]

        # 4. Resolution
        l_res = QHBoxLayout()
        make_tight(l_res)
        l_res.addWidget(QLabel("해상도:"))
        
        global_res = cfg.get('defaults', 'resolution') or 512
        saved_w = self.saved_config.get('inpaint_width', 0)
        saved_h = self.saved_config.get('inpaint_height', 0)
        
        self.spin_inpaint_w = QSpinBox(); self.spin_inpaint_w.setRange(0, 2048); 
        self.spin_inpaint_w.setValue(saved_w if saved_w > 0 else global_res)
        
        self.spin_inpaint_h = QSpinBox(); self.spin_inpaint_h.setRange(0, 2048);
        self.spin_inpaint_h.setValue(saved_h if saved_h > 0 else global_res)
        
        l_res.addWidget(self.spin_inpaint_w); l_res.addWidget(QLabel("x")); l_res.addWidget(self.spin_inpaint_h)
        l_inp.addLayout(l_res, 5, 0, 1, 3)
        
        # 5. Options (Merge, Color, Rotate)
        l_opts = QHBoxLayout()
        make_tight(l_opts)
        
        self.combo_mask_merge = QComboBox(); self.combo_mask_merge.addItems(["None", "Merge", "Merge+Inv"])
        saved_merge = self.saved_config.get('mask_merge_mode', "None")
        if isinstance(saved_merge, bool): saved_merge = "Merge" if saved_merge else "None"
        elif saved_merge == "Merge and Invert": saved_merge = "Merge+Inv" # Shorten for layout
        self.combo_mask_merge.setCurrentText(saved_merge)
        
        l_opts.addWidget(QLabel("병합:"))
        l_opts.addWidget(self.combo_mask_merge)
        
        self.combo_color_fix = QComboBox(); self.combo_color_fix.addItems(["None", "Wavelet", "Adain", "Histogram", "Linear"])
        self.combo_color_fix.setCurrentText(self.saved_config.get('color_fix', "None"))
        l_opts.addWidget(QLabel("색감:"))
        l_opts.addWidget(self.combo_color_fix)
        
        self.chk_auto_rotate = QCheckBox("자동 회전")
        self.chk_auto_rotate.setChecked(self.saved_config.get('auto_rotate', True))
        l_opts.addWidget(self.chk_auto_rotate)
        
        l_inp.addLayout(l_opts, 6, 0, 1, 3)
        
        # [Ref] Single Vertical Layout for Tab 2
        g_inp.setLayout(l_inp)
        t2_layout.addWidget(g_inp)
        
        # Group: Soft Inpainting

        # Group: Soft Inpainting
        self.g_soft = QGroupBox("Soft Inpainting")
        self.g_soft.setCheckable(True)
        self.g_soft.setChecked(bool(self.saved_config.get('use_soft_inpainting', False)))
        l_soft = QGridLayout()
        
        # Sliders
        def_soft_bias = cfg.get('defaults', 'soft_schedule_bias') or 1.0
        def_soft_pres = cfg.get('defaults', 'soft_preservation_strength') or 0.5
        def_soft_cont = cfg.get('defaults', 'soft_transition_contrast') or 4.0
        
        # Sliders (1-Column Stack for Clean Layout)
        # [Ref] Row 0: Bias
        self.add_slider_row(l_soft, 0, "Bias:", "soft_schedule_bias", 0.0, 8.0, 
                            self.saved_config.get('soft_schedule_bias', def_soft_bias), 0.1)
        # [Ref] Row 1: Preserve
        self.add_slider_row(l_soft, 1, "Pres:", "soft_preservation_strength", 0.0, 1.0, 
                            self.saved_config.get('soft_preservation_strength', def_soft_pres), 0.05)
                            
        # [Ref] Row 2: Contrast
        self.add_slider_row(l_soft, 2, "Cont:", "soft_transition_contrast", 1.0, 32.0, 
                            self.saved_config.get('soft_transition_contrast', def_soft_cont), 0.5)
                            
        l_soft.addWidget(QLabel("<b>Pixel Composite</b>"), 3, 0, 1, 3)
        
        def_soft_infl = cfg.get('defaults', 'soft_mask_influence') or 0.0
        def_soft_diff = cfg.get('defaults', 'soft_diff_threshold') or 0.5
        def_soft_dcont = cfg.get('defaults', 'soft_diff_contrast') or 2.0
        
        # [Ref] Row 4: Mask Infl
        self.add_slider_row(l_soft, 4, "Infl:", "soft_mask_influence", 0.0, 1.0, 
                            self.saved_config.get('soft_mask_influence', def_soft_infl), 0.05)
        # [Ref] Row 5: Diff Thresh
        self.add_slider_row(l_soft, 5, "Diff:", "soft_diff_threshold", 0.0, 1.0, 
                            self.saved_config.get('soft_diff_threshold', def_soft_diff), 0.05)
                            
        # [Ref] Row 6: Diff Cont
        self.add_slider_row(l_soft, 6, "Cont:", "soft_diff_contrast", 0.0, 8.0, 
                            self.saved_config.get('soft_diff_contrast', def_soft_dcont), 0.1)
                            
        l_soft.setVerticalSpacing(2)
        l_soft.setColumnStretch(1, 1) # [Fix]
        self.g_soft.setLayout(l_soft)
        t2_layout.addWidget(self.g_soft)
        
        # Group: ControlNet
        g_cn = QGroupBox("ControlNet")
        l_cn = QGridLayout()
        
        self.combo_cn_model = QComboBox(); self.combo_cn_model.addItem("None")
        cn_dir = cfg.get_path('controlnet')
        if cn_dir and os.path.exists(cn_dir):
            try:
                cn_models = [f for f in os.listdir(cn_dir) if f.endswith(('.pth', '.safetensors', '.bin'))]
                self.combo_cn_model.addItems(sorted(cn_models))
            except: pass
        
        default_cn = cfg.get('files', 'controlnet_tile')
        saved_cn = self.saved_config.get('control_model', default_cn or "None")
        self.combo_cn_model.setCurrentText(saved_cn)
        
        self.combo_cn_module = QComboBox() 
        self.combo_cn_module.addItems(["None", "openpose", "canny", "depth_midas", "openpose_full", "softedge_pidinet", "scribble_hed"])
        self.combo_cn_module.setCurrentText(self.saved_config.get('control_module', "None"))

        l_cn.addWidget(QLabel("모델:"), 0, 0)
        l_cn.addWidget(self.combo_cn_model, 0, 1)
        l_cn.addWidget(QLabel("전처리:"), 1, 0)
        l_cn.addWidget(self.combo_cn_module, 1, 1)
        
        def_cn_weight = cfg.get('defaults', 'controlnet_weight') or 1.0
        # [Ref] 1-Column Grid for ControlNet
        # Row 2: Weight
        self.add_slider_row(l_cn, 2, "Wgt:", "control_weight", 0.0, 2.0, def_cn_weight, 0.1)
        # Row 3: Start
        self.add_slider_row(l_cn, 3, "Start:", "guidance_start", 0.0, 1.0, 0.0, 0.05)
        
        # Row 4: End
        self.add_slider_row(l_cn, 4, "End:", "guidance_end", 0.0, 1.0, 1.0, 0.05)
        
        l_cn.setVerticalSpacing(2)
        l_cn.setColumnStretch(1, 1) # [Fix]
        g_cn.setLayout(l_cn)
        t2_layout.addWidget(g_cn)
        # Final stretch for the whole tab
        t2_layout.addStretch()
        
        # [Ref] Direct Vertical Stack - No Splitter
        # t2_layout.addWidget(left_widget, 1) 
        # t2_layout.addWidget(right_widget, 1)
        
        self.tabs.addTab(tab2, "인페인팅 (Inpaint)")
        
        # --- TAB 3: BMAB & Composition ---
        tab3 = QWidget()
        t3_layout = QVBoxLayout(tab3)
        
        # BMAB Preprocess
        g_bmab = QGroupBox("BMAB 이미지 보정 (Basic)")
        l_bmab_main = QVBoxLayout() 

        self.chk_bmab_enabled = QCheckBox("활성화 (Enable)")
        def_bmab_en = cfg.get('defaults', 'bmab_enabled')
        if def_bmab_en is None: def_bmab_en = True
        self.chk_bmab_enabled.setChecked(self.saved_config.get('bmab_enabled', def_bmab_en))
        l_bmab_main.addWidget(self.chk_bmab_enabled)

        # [Ref] BMAB Basic Tab Layout (Single Column for 460px)
        l_bmab_grid = QGridLayout()
        
        # Defaults
        def_con = cfg.get('defaults', 'bmab_contrast') or 1.0
        def_bri = cfg.get('defaults', 'bmab_brightness') or 1.0
        def_sha = cfg.get('defaults', 'bmab_sharpness') or 1.0
        def_sat = cfg.get('defaults', 'bmab_color_saturation') or 1.0
        def_temp= cfg.get('defaults', 'bmab_color_temperature') or 0.0
        def_na  = cfg.get('defaults', 'bmab_noise_alpha') or 0.0
        def_naf = cfg.get('defaults', 'bmab_noise_alpha_final') or 0.0
        
        # Column 1
        # Contrast: 0-2 (Default 1)
        self.add_slider_row(l_bmab_grid, 0, "대비 (Contrast):", "bmab_contrast", 0.0, 2.0, 
                            self.saved_config.get('bmab_contrast', def_con), 0.05, start_col=0)
        # Brightness: 0-2 (Default 1)
        self.add_slider_row(l_bmab_grid, 1, "밝기 (Brightness):", "bmab_brightness", 0.0, 2.0, 
                            self.saved_config.get('bmab_brightness', def_bri), 0.05, start_col=0)
        # Sharpness: -5 to 5 (Default 1) - Ref says 1 default? Code says 1.
        self.add_slider_row(l_bmab_grid, 2, "선명도 (Sharpness):", "bmab_sharpness", -5.0, 5.0, 
                            self.saved_config.get('bmab_sharpness', def_sha), 0.1, start_col=0)
        # Color (Saturation): 0-2 (Default 1)
        self.add_slider_row(l_bmab_grid, 3, "채도 (Color):", "bmab_color_saturation", 0.0, 2.0, 
                            self.saved_config.get('bmab_color_saturation', def_sat), 0.01, start_col=0)

        # Continues in single column (Rows 4, 5, 6)
        # Color Temp: -2000 to +2000 (Default 0)
        self.add_slider_row(l_bmab_grid, 4, "색온도 (Temp):", "bmab_color_temperature", -2000.0, 2000.0, 
                            self.saved_config.get('bmab_color_temperature', def_temp), 10.0, start_col=0)
        # Noise Alpha: 0-1 (Default 0)
        self.add_slider_row(l_bmab_grid, 5, "노이즈 (Alpha):", "bmab_noise_alpha", 0.0, 1.0, 
                            self.saved_config.get('bmab_noise_alpha', def_na), 0.01, start_col=0)
        # Noise Alpha Final: 0-1 (Default 0)
        self.add_slider_row(l_bmab_grid, 6, "노이즈 (Final):", "bmab_noise_alpha_final", 0.0, 1.0, 
                            self.saved_config.get('bmab_noise_alpha_final', def_naf), 0.01, start_col=0)
        l_bmab_grid.setColumnStretch(1, 1) # [Fix]
        
        l_bmab_main.addLayout(l_bmab_grid)
        
        # Edge Enhancement (Kept separate as it was in "Edge" tab in ref, 
        # but user might still want it here. Let's keep it but slightly separated or in a sub-group)
        self.g_edge = QGroupBox("엣지 강화 (Edge)")
        self.g_edge.setCheckable(True)
        def_edge_en = cfg.get('defaults', 'bmab_edge_enabled')
        if def_edge_en is None: def_edge_en = False
        self.g_edge.setChecked(bool(self.saved_config.get('bmab_edge_enabled', def_edge_en))) # Collapsed/Disabled by default to match clean Basic tab look
        l_edge = QGridLayout()
        
        def_edge_str = cfg.get('defaults', 'bmab_edge_strength') or 0.0
        def_edge_low = cfg.get('defaults', 'bmab_edge_low') or 50
        def_edge_high = cfg.get('defaults', 'bmab_edge_high') or 200
        
        self.add_slider_row(l_edge, 0, "강도:", "bmab_edge_strength", 0.0, 1.0, 
                            self.saved_config.get('bmab_edge_strength', def_edge_str), 0.05)
        self.add_slider_row(l_edge, 1, "Low:", "bmab_edge_low", 0, 255, 
                            self.saved_config.get('bmab_edge_low', def_edge_low), 1)
        self.add_slider_row(l_edge, 2, "High:", "bmab_edge_high", 0, 255, 
                            self.saved_config.get('bmab_edge_high', def_edge_high), 1)
        l_edge.setColumnStretch(1, 1) # [Fix]
        self.g_edge.setLayout(l_edge)
        l_bmab_main.addWidget(self.g_edge)

        g_bmab.setLayout(l_bmab_main)
        t3_layout.addWidget(g_bmab)
        
        # Composition
        g_comp = QGroupBox("캔버스 확장 (Resize by Person)")
        l_comp = QGridLayout()
        self.chk_resize_enable = QCheckBox("활성화")
        def_resize_en = cfg.get('defaults', 'resize_enable')
        if def_resize_en is None: def_resize_en = False
        self.chk_resize_enable.setChecked(self.saved_config.get('resize_enable', def_resize_en))
        l_comp.addWidget(self.chk_resize_enable, 0, 0)
        
        def_resize_ratio = cfg.get('defaults', 'resize_ratio') or 0.6
        self.add_slider_row(l_comp, 1, "목표 비율:", "resize_ratio", 0.1, 1.0, 
                            self.saved_config.get('resize_ratio', def_resize_ratio), 0.05)
        
        self.combo_resize_align = QComboBox(); self.combo_resize_align.addItems(["Center", "Bottom", "Top"])
        def_resize_align = cfg.get('defaults', 'resize_align') or "Center"
        self.combo_resize_align.setCurrentText(self.saved_config.get('resize_align', def_resize_align))
        l_comp.addWidget(QLabel("정렬:"), 2, 0)
        l_comp.addWidget(self.combo_resize_align, 2, 1)
        
        # [New] Landscape Detail Feature
        self.chk_landscape_detail = QCheckBox("풍경 속 인물 디테일링 (Landscape)")
        self.chk_landscape_detail.setToolTip("활성화 시, 인물이 작아도(최소 크기 미달이어도) 강제로 디테일링을 수행합니다.")
        def_land = cfg.get('defaults', 'bmab_landscape_detail')
        if def_land is None: def_land = False
        self.chk_landscape_detail.setChecked(self.saved_config.get('bmab_landscape_detail', def_land))
        l_comp.addWidget(self.chk_landscape_detail, 3, 0, 1, 2)
        l_comp.setColumnStretch(1, 1) # [Fix]
        
        g_comp.setLayout(l_comp)
        t3_layout.addWidget(g_comp)
        t3_layout.addStretch()
        self.tabs.addTab(tab3, "BMAB (Effect)")

        # --- TAB 4: Detail Daemon --- 
        tab_dd = QWidget()
        t_dd_layout = QVBoxLayout(tab_dd)
        
        # ---------------------------------------------------------
        # [New] Detail Daemon Group (Matching Screenshot)
        # ---------------------------------------------------------
        g_dd = QGroupBox("Detail Daemon")
        l_dd = QVBoxLayout()
        l_dd.setSpacing(15)
        
        # Top Row: Active & Hires Pass
        l_dd_top = QHBoxLayout()
        self.chk_dd_active = QCheckBox("활성화 (Enable)")
        self.chk_dd_active.setChecked(self.saved_config.get('dd_enabled', False)) # Restore State
        self.chk_dd_hires = QCheckBox("Hires Pass") # Not fully implemented logic-wise yet, but UI placeholder
        l_dd_top.addWidget(self.chk_dd_active)
        l_dd_top.addStretch()
        l_dd_top.addWidget(self.chk_dd_hires)
        l_dd.addLayout(l_dd_top)
        
        # Sliders Grid
        l_dd_grid = QGridLayout()
        l_dd_grid.setSpacing(10)
        
        # Helper to create slider row
        def create_slider(label, min_val, max_val, step, default_val):
            l_row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(80)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * 100), int(max_val * 100)) # Scaled by 100
            slider.setValue(int(default_val * 100))
            
            val_lbl = QLabel(f"{default_val}")
            val_lbl.setMinimumWidth(40)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            # Sync
            def on_change(val):
                float_val = val / 100.0
                if step < 0.1: # 0.01
                     val_lbl.setText(f"{float_val:.2f}")
                else:
                     val_lbl.setText(f"{float_val:.1f}")
            
            slider.valueChanged.connect(on_change)
            
            l_row.addWidget(lbl)
            l_row.addWidget(slider)
            l_row.addWidget(val_lbl)
            return l_row, slider # Return layout and slider object

        # [Modified] Layout: Vertical Stack for 460px width
        l_middle_hbox = QVBoxLayout()
        
        # Left Side: Sliders
        l_sliders_vbox = QVBoxLayout()
        
        # Row 1: Amount
        # Row 1: Amount
        def_dd_amt = cfg.get('defaults', 'dd_amount') or 0.1
        l_amt, self.slider_dd_amount = create_slider("Detail Amount", -1.0, 1.0, 0.01, self.saved_config.get('dd_amount', def_dd_amt))
        l_sliders_vbox.addLayout(l_amt)
        
        # Grid Controls: Start, End, Start Offset, End Offset, Bias, Exponent
        l_dd_grid = QGridLayout()
        l_dd_grid.setSpacing(10)

        # Row 2
        def_dd_start = cfg.get('defaults', 'dd_start') or 0.2
        def_dd_end = cfg.get('defaults', 'dd_end') or 0.8
        l_st, self.slider_dd_start = create_slider("Start", 0.0, 1.0, 0.01, self.saved_config.get('dd_start', def_dd_start))
        l_ed, self.slider_dd_end = create_slider("End", 0.0, 1.0, 0.01, self.saved_config.get('dd_end', def_dd_end))
        l_dd_grid.addLayout(l_st, 0, 0)
        l_dd_grid.addLayout(l_ed, 0, 1)

        # Row 3
        def_dd_st_off = cfg.get('defaults', 'dd_start_offset') or 0.0
        def_dd_ed_off = cfg.get('defaults', 'dd_end_offset') or 0.0
        l_st_off, self.slider_dd_start_offset = create_slider("Start Offset", -1.0, 1.0, 0.01, self.saved_config.get('dd_start_offset', def_dd_st_off))
        l_ed_off, self.slider_dd_end_offset = create_slider("End Offset", -1.0, 1.0, 0.01, self.saved_config.get('dd_end_offset', def_dd_ed_off))
        l_dd_grid.addLayout(l_st_off, 1, 0)
        l_dd_grid.addLayout(l_ed_off, 1, 1)
        
        # Row 4
        def_dd_bias = cfg.get('defaults', 'dd_bias') or 0.5
        def_dd_exp = cfg.get('defaults', 'dd_exponent') or 1.0
        l_bias, self.slider_dd_bias = create_slider("Bias", 0.0, 1.0, 0.01, self.saved_config.get('dd_bias', def_dd_bias))
        l_exp, self.slider_dd_exponent = create_slider("Exponent", 0.0, 10.0, 0.05, self.saved_config.get('dd_exponent', def_dd_exp))
        l_dd_grid.addLayout(l_bias, 2, 0)
        l_dd_grid.addLayout(l_exp, 2, 1)
        
        # Row 5 (Fade)
        def_dd_fade = cfg.get('defaults', 'dd_fade') or 0.0
        l_fade, self.slider_dd_fade = create_slider("Fade", 0.0, 1.0, 0.05, self.saved_config.get('dd_fade', def_dd_fade))
        l_dd_grid.addLayout(l_fade, 3, 0)
        
        l_sliders_vbox.addLayout(l_dd_grid)
        l_sliders_vbox.addStretch()
        l_middle_hbox.addLayout(l_sliders_vbox, stretch=6) # Sliders take 60%
        
        # Right Side: Graph & Smooth Checkbox
        l_right_vbox = QVBoxLayout()
        self.dd_graph = ScheduleGraphWidget()
        l_right_vbox.addWidget(self.dd_graph)
        
        # Smooth Checkbox (Bottom of Graph)
        self.chk_dd_smooth = QCheckBox("Smooth")
        def_dd_smooth = cfg.get('defaults', 'dd_smooth')
        if def_dd_smooth is None: def_dd_smooth = True
        self.chk_dd_smooth.setChecked(self.saved_config.get('dd_smooth', def_dd_smooth))
        l_right_vbox.addWidget(self.chk_dd_smooth)
        
        l_middle_hbox.addLayout(l_right_vbox, stretch=4) # Graph takes 40%
        
        l_dd.addLayout(l_middle_hbox)
        
        # Connect Signals for Graph Update
        self.slider_dd_amount.valueChanged.connect(self.update_dd_graph)
        self.slider_dd_start.valueChanged.connect(self.update_dd_graph)
        self.slider_dd_end.valueChanged.connect(self.update_dd_graph)
        self.slider_dd_start_offset.valueChanged.connect(self.update_dd_graph)
        self.slider_dd_end_offset.valueChanged.connect(self.update_dd_graph)
        self.slider_dd_bias.valueChanged.connect(self.update_dd_graph)
        self.slider_dd_exponent.valueChanged.connect(self.update_dd_graph)
        self.slider_dd_fade.valueChanged.connect(self.update_dd_graph)
        self.chk_dd_smooth.stateChanged.connect(self.update_dd_graph)
        
        # Initial Update
        self.update_dd_graph()

        # Advanced "More Knobs" (Collapsible or just Group)
        # Screenshot shows "More Knobs:" with numeric inputs and Mode.
        gb_more = QGroupBox("More Knobs (Numeric Input)")
        gb_more.setCheckable(True)
        gb_more.setChecked(False)  # Collapsed by default (via uncheck)
        
        l_more = QGridLayout()
        
        # Numeric Inputs (SpinBoxes)
        self.spin_dd_amount = QDoubleSpinBox(); self.spin_dd_amount.setRange(-2.0, 2.0); self.spin_dd_amount.setSingleStep(0.01); self.spin_dd_amount.setValue(self.saved_config.get('dd_amount', def_dd_amt))
        self.spin_dd_st_off = QDoubleSpinBox(); self.spin_dd_st_off.setRange(-2.0, 2.0); self.spin_dd_st_off.setSingleStep(0.01); self.spin_dd_st_off.setValue(self.saved_config.get('dd_start_offset', def_dd_st_off))
        self.spin_dd_ed_off = QDoubleSpinBox(); self.spin_dd_ed_off.setRange(-2.0, 2.0); self.spin_dd_ed_off.setSingleStep(0.01); self.spin_dd_ed_off.setValue(self.saved_config.get('dd_end_offset', def_dd_ed_off))
        
        l_more.addWidget(QLabel("Amount"), 0, 0); l_more.addWidget(self.spin_dd_amount, 1, 0)
        l_more.addWidget(QLabel("Start Offset"), 0, 1); l_more.addWidget(self.spin_dd_st_off, 1, 1)
        l_more.addWidget(QLabel("End Offset"), 0, 2); l_more.addWidget(self.spin_dd_ed_off, 1, 2)
        
        # Mode Dropdown
        self.combo_dd_mode = QComboBox()
        self.combo_dd_mode.addItems(["both", "cond", "uncond"])
        def_dd_mode = cfg.get('defaults', 'dd_mode') or "both"
        self.combo_dd_mode.setCurrentText(self.saved_config.get('dd_mode', def_dd_mode))
        
        l_more.addWidget(QLabel("모드 (Mode)"), 0, 3); l_more.addWidget(self.combo_dd_mode, 1, 3)
        
        gb_more.setLayout(l_more)
        l_dd.addWidget(gb_more)
        
        # Two-way binding for Sliders <-> SpinBoxes (Optional polish)
        self.spin_dd_amount.valueChanged.connect(lambda v: self.slider_dd_amount.setValue(int(v*100)))
        self.slider_dd_amount.valueChanged.connect(lambda v: self.spin_dd_amount.setValue(v/100.0))
        
        self.spin_dd_st_off.valueChanged.connect(lambda v: self.slider_dd_start_offset.setValue(int(v*100)))
        self.slider_dd_start_offset.valueChanged.connect(lambda v: self.spin_dd_st_off.setValue(v/100.0))
 
        self.spin_dd_ed_off.valueChanged.connect(lambda v: self.slider_dd_end_offset.setValue(int(v*100)))
        self.slider_dd_end_offset.valueChanged.connect(lambda v: self.spin_dd_ed_off.setValue(v/100.0))
        
        g_dd.setLayout(l_dd)
        t_dd_layout.addWidget(g_dd)
        t_dd_layout.addStretch()
        
        self.tabs.addTab(tab_dd, "Detail Daemon")
        # [Fix] Default to Detection Tab (User Request)
        # self.tabs.setCurrentWidget(tab_dd) 
        self.tabs.setCurrentIndex(0) # 0 = Detect

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
        self.combo_sep_sampler = QComboBox(); self.combo_sep_sampler.addItems([
            "Euler a", "Euler", "DPM++ 2M", "DPM++ SDE", "DDIM", "UniPC"
        ])
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
        l_adv.setColumnStretch(1, 1) # [Fix] Expand combo column
        l_adv.setColumnStretch(2, 1) # Expand scheduler too
        
        # Steps/CFG/Clip (Split into two rows for 1-column compatibility)
        l_sub1 = QHBoxLayout(); make_tight(l_sub1)
        self.chk_sep_steps=QCheckBox("Steps"); self.chk_sep_steps.setChecked(self.saved_config.get('sep_steps', False))
        self.spin_sep_steps=QSpinBox(); self.spin_sep_steps.setValue(self.saved_config.get('steps', 20))
        self.chk_sep_cfg=QCheckBox("CFG"); self.chk_sep_cfg.setChecked(self.saved_config.get('sep_cfg', False))
        self.spin_sep_cfg=QDoubleSpinBox(); self.spin_sep_cfg.setValue(self.saved_config.get('cfg_scale', 7.0))
        l_sub1.addWidget(self.chk_sep_steps); l_sub1.addWidget(self.spin_sep_steps)
        l_sub1.addWidget(self.chk_sep_cfg); l_sub1.addWidget(self.spin_sep_cfg)
        l_adv.addLayout(l_sub1, 2, 0, 1, 4)
        
        l_sub2 = QHBoxLayout(); make_tight(l_sub2)
        self.chk_sep_clip=QCheckBox("Clip"); self.chk_sep_clip.setChecked(self.saved_config.get('sep_clip', False))
        self.spin_clip=QSpinBox(); self.spin_clip.setRange(1,12); self.spin_clip.setValue(self.saved_config.get('clip_skip', 2))
        l_sub2.addWidget(self.chk_sep_clip); l_sub2.addWidget(self.spin_clip)
        self.chk_sep_noise = QCheckBox("Sep Noise"); self.chk_sep_noise.setChecked(self.saved_config.get('sep_noise', False))
        self.chk_restore_face = QCheckBox("Restore Face"); self.chk_restore_face.setChecked(self.saved_config.get('restore_face', False))
        l_sub2.addWidget(self.chk_sep_noise)
        l_sub2.addWidget(self.chk_restore_face)
        l_adv.addLayout(l_sub2, 3, 0, 1, 4)
        
        # [New] Restore Strength Slider (Row 4)
        def_restore = cfg.get('defaults', 'restore_face_strength') or 1.0
        self.add_slider_row(l_adv, 4, "얼굴복원강도:", "restore_face_strength", 0.0, 1.0, 
                            self.saved_config.get('restore_face_strength', def_restore), 0.05, start_col=0)
        
        # [New] Post Sharpen Slider (Row 5)
        def_sharpen = cfg.get('defaults', 'post_sharpen') or 0.3
        self.add_slider_row(l_adv, 5, "후처리 선명도:", "post_sharpen", 0.0, 1.5, 
                            self.saved_config.get('post_sharpen', def_sharpen), 0.05, start_col=0)
        
        g_adv.setLayout(l_adv)
        t4_layout.addWidget(g_adv)

        # [New] Hires Fix (Reforge Style) Group
        g_hires = QGroupBox("Hires Fix (Reforge Style)")
        l_hires = QGridLayout()
        
        # Row 0: Enable Checkbox
        self.chk_hires = QCheckBox("Enable Hires Fix")
        self.chk_hires.setChecked(self.saved_config.get('use_hires_fix', False))
        l_hires.addWidget(self.chk_hires, 0, 0, 1, 4)
        
        # Row 1: Upscaler (Dynamic Load)
        l_hires.addWidget(QLabel("Upscaler"), 1, 0)
        self.combo_hires_upscaler = QComboBox()
        self.combo_hires_upscaler.addItem("None")
        # Dynamic Load from D:\AI_Models\ESRGAN
        esrgan_path = r"D:\AI_Models\ESRGAN"
        if os.path.exists(esrgan_path):
            try:
                models = [f for f in os.listdir(esrgan_path) if f.casefold().endswith('.pth')]
                self.combo_hires_upscaler.addItems(models)
            except Exception as e:
                print(f"[UI] Error loading ESRGAN models: {e}")
        self.combo_hires_upscaler.setCurrentText(self.saved_config.get('hires_upscaler', 'None'))
        l_hires.addWidget(self.combo_hires_upscaler, 1, 1, 1, 3)
        
        # Row 2: Upscale Factor (Slider)
        def_hires_up = cfg.get('defaults', 'hires_upscale_factor') or 1.5
        self.add_slider_row(l_hires, 2, "Upscale Factor", 'hires_upscale_factor', 1.0, 4.0, 
                            def_hires_up, 0.05)
                            
        # Row 3: Hires Steps, Denoise
        def_hires_steps = cfg.get('defaults', 'hires_steps') or 14
        def_hires_denoise = cfg.get('defaults', 'hires_denoise') or 0.4
        
        self.add_slider_row(l_hires, 3, "Hires Steps", 'hires_steps', 0, 50, 
                            def_hires_steps, 1)
        self.add_slider_row(l_hires, 4, "Denoise Strength", 'hires_denoise', 0.01, 1.0, 
                            def_hires_denoise, 0.01)
        l_hires.setColumnStretch(1, 1) # [Fix] Stretch slider

        # Row 4: Width/Height Override (0=Auto)
        l_hires.addWidget(QLabel("Width (0=Auto)"), 5, 0)
        self.spin_hires_w = QSpinBox()
        self.spin_hires_w.setRange(0, 4096)
        self.spin_hires_w.setValue(self.saved_config.get('hires_width', 0))
        self.spin_hires_w.setSingleStep(8)
        self.settings['hires_width'] = {
            'widget': self.spin_hires_w,
            'default': 0
        } # Hook for get_config
        l_hires.addWidget(self.spin_hires_w, 5, 1)
        
        l_hires.addWidget(QLabel("Height (0=Auto)"), 5, 2)
        self.spin_hires_h = QSpinBox()
        self.spin_hires_h.setRange(0, 4096)
        self.spin_hires_h.setValue(self.saved_config.get('hires_height', 0))
        self.spin_hires_h.setSingleStep(8)
        self.settings['hires_height'] = {
            'widget': self.spin_hires_h,
            'default': 0
        }
        l_hires.addWidget(self.spin_hires_h, 5, 3)

        # Row 5: Hires CFG
        def_hires_cfg = cfg.get('defaults', 'hires_cfg') or 5.0
        self.add_slider_row(l_hires, 6, "Hires CFG Scale", 'hires_cfg', 1.0, 30.0, 
                            def_hires_cfg, 0.5)

        g_hires.setLayout(l_hires)
        t4_layout.addWidget(g_hires)
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
        # [Fix] Fixed Label Width for consistent alignment (Increased to 130px for bilingual labels)
        label.setFixedWidth(130) 
        
        slider = QSlider(Qt.Orientation.Horizontal)
        # [Fix] Expand slider to fill space
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
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
        # [Fix] Fixed Spinbox Width (User Requested 70px)
        spin.setFixedWidth(70) 
        spin.setObjectName("clean_spin")
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        spin.setValue(loaded_val)
        slider.setValue(int(loaded_val * scale))
        
        if is_float:
            slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        else:
            slider.valueChanged.connect(lambda v: spin.setValue(int(v / scale)))
            
        # Add to layout
        # [Ref] Revert to simple 3-column row (Label, Slider, Spinbox)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(spin, row, 2)
            
        # 스핀박스 -> 슬라이더 (값 변경 시 즉시 반영)
        # 스핀박스 -> 슬라이더 (값 변경 시 즉시 반영)
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * scale)))
        
        # Note: addWidget called above, don't duplicate. 
        # The original code had duplicate addWidget calls? 
        # Yes, lines 1004-1006 were duplicates of 997-999 in previous context.
        # I will remove them here by replacing with empty/nothing if I am encompassing them.
        # My target block ends at 1006, so I am replacing the duplication too.
        # Wait, I should double check if I am breaking grid logic.
        # The previous code:
        # 997: layout.addWidget...
        # ...
        # 1004: layout.addWidget... (Duplicate!)
        # So I will just NOT include the duplicate widgets.
        
        self.settings[key] = {
            'widget': spin,
            'default': default_val
        }

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

    
    def on_detect_preview_clicked(self):
        """탐지 버튼 클릭 시 현재 설정으로 탐지 미리보기 요청"""
        cfg = self.get_config()
        self.preview_requested.emit(cfg)

    def on_reset_clicked(self):
        """현재 패스의 모든 설정을 기본값으로 초기화"""
        from PyQt6.QtWidgets import QMessageBox
        # Confirm
        ret = QMessageBox.question(self, "설정 초기화", f"'{self.unit_name}'의 모든 설정을 초기화하시겠습니까?", 
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret == QMessageBox.StandardButton.No: return

        # 1. Reset Sliders/Spinboxes (Dictionary stored)
        for key, data in self.settings.items():
            widget = data['widget']
            def_val = data['default']
            
            # Special Handling for SpinBox ranges or Ratio conversions?
            # add_slider_row already took care of range setup using default_val context?
            # No, 'default' stored is the raw default passed to add_slider_row.
            # Spinboxes expect raw values (e.g. 50 for 50%).
            widget.setValue(def_val)

        # 2. Reset Explicit Widgets
        # Basic
        self.chk_enable.setChecked("1" in self.unit_name) # Pass 1 enabled by default, others false
        self.radio_yolo.setChecked(True)
        self.chk_auto_prompt.setChecked(True)
        
        # Model & Prompts
        default_model = cfg.get('files', 'sam_file') or "face_yolov8n.pt"
        idx = self.combo_model.findText(default_model)
        if idx >= 0: self.combo_model.setCurrentIndex(idx)
        
        self.txt_yolo_classes.setText("")
        self.txt_pos.setText("")
        self.txt_neg.setText("")
        
        # Detection
        self.combo_gender.setCurrentText("All")
        self.chk_ignore_edge.setChecked(False)
        self.chk_anatomy.setChecked(True)
        self.chk_pose_rotation.setChecked(False)
        self.spin_top_k.setValue(20) # Default max_det
        self.radio_sort_conf.setChecked(True) # Default Sort: Confidence
        
        # Inpaint
        # self.spin_inpaint_w.setValue(512) # Handled by apply_model_presets usually, or global default?
        global_res = cfg.get('defaults', 'resolution') or 512
        self.spin_inpaint_w.setValue(global_res)
        self.spin_inpaint_h.setValue(global_res)
        
        self.combo_mask_merge.setCurrentText("None")
        # [Fix] Reset Mask Content & Area
        self.radio_content_orig.setChecked(True)
        self.radio_area_masked.setChecked(True)
        self.chk_auto_rotate.setChecked(True)
        self.combo_color_fix.setCurrentText("None")
        
        # [Fix] Reset New Features
        self.chk_landscape_detail.setChecked(False)
        # Soft Inpainting & Edge Group (Instance Variables)
        if hasattr(self, 'g_soft'): self.g_soft.setChecked(False)
        if hasattr(self, 'g_edge'): self.g_edge.setChecked(False)
            
        # ControlNet
        self.combo_cn_model.setCurrentIndex(0) # None
        self.combo_cn_module.setCurrentIndex(0) # None
        
        # Detail Daemon
        self.chk_dd_active.setChecked(False)
        self.chk_dd_hires.setChecked(False)
        self.chk_dd_smooth.setChecked(True)
        self.combo_dd_mode.setCurrentText("both")
        
        # [Fix] Manual Reset for DD Sliders (not in self.settings)
        # We set sliders, which syncs to spinboxes via signal
        def_dd_amt = cfg.get('defaults', 'dd_amount') or 0.1
        def_dd_start = cfg.get('defaults', 'dd_start') or 0.2
        def_dd_end = cfg.get('defaults', 'dd_end') or 0.8
        def_dd_st_off = cfg.get('defaults', 'dd_start_offset') or 0.0
        def_dd_ed_off = cfg.get('defaults', 'dd_end_offset') or 0.0
        def_dd_bias = cfg.get('defaults', 'dd_bias') or 0.5
        def_dd_exp = cfg.get('defaults', 'dd_exponent') or 1.0
        def_dd_fade = cfg.get('defaults', 'dd_fade') or 0.0
        
        self.slider_dd_amount.setValue(int(def_dd_amt * 100))
        self.slider_dd_start.setValue(int(def_dd_start * 100))
        self.slider_dd_end.setValue(int(def_dd_end * 100))
        self.slider_dd_start_offset.setValue(int(def_dd_st_off * 100))
        self.slider_dd_end_offset.setValue(int(def_dd_ed_off * 100))
        self.slider_dd_bias.setValue(int(def_dd_bias * 100))
        self.slider_dd_exponent.setValue(int(def_dd_exp * 100))
        self.slider_dd_fade.setValue(int(def_dd_fade * 100))
        
        # BMAB
        self.chk_bmab_enabled.setChecked(True)
        
        # Composition
        self.chk_resize_enable.setChecked(False)
        self.combo_resize_align.setCurrentText("Center")
        
        # Advanced (Overrides)
        self.chk_sep_ckpt.setChecked(False)
        self.chk_sep_vae.setChecked(False)
        self.chk_sep_sampler.setChecked(False)
        
        self.chk_sep_steps.setChecked(False)
        self.spin_sep_steps.setValue(20) # Default
        
        self.chk_sep_cfg.setChecked(False)
        self.spin_sep_cfg.setValue(7.0) # Default
        
        self.chk_sep_clip.setChecked(False)
        self.spin_clip.setValue(2) # Default
        
        self.chk_sep_noise.setChecked(False)
        
        self.chk_restore_face.setChecked(False)
        self.chk_hires.setChecked(False)
        self.combo_hires_upscaler.setCurrentText("None")
        self.spin_hires_w.setValue(0)
        self.spin_hires_h.setValue(0)
        
        # Update UI state (enable/disable based on checks)
        # Toggling checkboxes triggers signals usually, so separate update might not be needed?
        # Manually triggering crucial update logic if necessary.
        
        self.on_global_model_changed(self.combo_model.currentText()) # Re-apply resolution presets logic


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

    def update_dd_graph(self):
        """Update the Detail Daemon Graph based on current UI values."""
        # Visualize with 100 steps
        steps = 100
        start = self.slider_dd_start.value() / 100.0
        end = self.slider_dd_end.value() / 100.0
        bias = self.slider_dd_bias.value() / 100.0
        amount = self.slider_dd_amount.value() / 100.0 # Graph expects raw amount? make_schedule uses it.
        exponent = self.slider_dd_exponent.value() / 100.0
        start_offset = self.slider_dd_start_offset.value()
        end_offset = self.slider_dd_end_offset.value()
        fade = self.slider_dd_fade.value() / 100.0
        smooth = self.chk_dd_smooth.isChecked()
        
        # Generate Curve
        # make_schedule returns array of length 'steps'
        try:
            values = make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth)
            self.dd_graph.set_data(values)
        except Exception as e:
            print(f"Graph Error: {e}")

    def set_theme(self, mode):
        """ 테마 변경 시 호출됨 (그래프 등 커스텀 위젯 업데이트) """
        if hasattr(self, 'dd_graph'):
            self.dd_graph.set_theme(mode)

    def get_config(self):
        """Configs.py Key 완전 일치 및 누락 기능 복구 완료"""
        cfg = {
            'enabled': self.chk_enable.isChecked(),
            'detector_model': self.combo_model.currentText(),
            'yolo_classes': self.txt_yolo_classes.toPlainText(),
            'use_sam': self.radio_sam.isChecked(),
            
            # [Collect Config]
            'pos_prompt': self.txt_pos.toPlainText(),
            'neg_prompt': self.txt_neg.toPlainText(),
            'auto_prompt_injection': self.chk_auto_prompt.isChecked(),

            # --- Detection ---
            'gender_filter': self.combo_gender.currentText(),
            'ignore_edge_touching': self.chk_ignore_edge.isChecked(),
            'anatomy_check': self.chk_anatomy.isChecked(),
            'use_pose_rotation': self.chk_pose_rotation.isChecked(),
            'max_det': self.spin_top_k.value(),

            # --- Inpaint ---
            'inpaint_width': self.spin_inpaint_w.value(),
            'inpaint_height': self.spin_inpaint_h.value(),
            'mask_merge_mode': self.combo_mask_merge.currentText(),
            'use_noise_mask': False, # Deprecated/Merged into Mask Content
            'auto_rotate': self.chk_auto_rotate.isChecked(),
            'color_fix': self.combo_color_fix.currentText(),
            
            # [New] Mask Content logic
            'mask_content': 'original', # Default
            'inpaint_full_res': self.radio_area_whole.isChecked(),
            
            # [New] Soft Inpainting
            'use_soft_inpainting': self.g_soft.isChecked(),
            
            # Check radios for mask content string
            # (Assuming one is always checked)
            # Find which radio is checked without explicit variable references if possible, 
            # but we have member variables.
            'mask_content': 
                'fill' if self.radio_content_fill.isChecked() else
                'latent_noise' if self.radio_content_noise.isChecked() else
                'latent_nothing' if self.radio_content_nothing.isChecked() else
                'original',

            
            # --- ControlNet ---
            'control_model': self.combo_cn_model.currentText(),
            'control_module': self.combo_cn_module.currentText(),

            # [Detail Daemon Config]
            'dd_enabled': self.chk_dd_active.isChecked(),
            'dd_amount': self.spin_dd_amount.value(), # Use spinbox value (synced)
            'dd_start': float(self.slider_dd_start.value()) / 100.0,
            'dd_end': float(self.slider_dd_end.value()) / 100.0,
            'dd_start_offset': self.spin_dd_st_off.value(),
            'dd_end_offset': self.spin_dd_ed_off.value(),
            'dd_bias': float(self.slider_dd_bias.value()) / 100.0,
            'dd_exponent': float(self.slider_dd_exponent.value()) / 100.0,
            'dd_fade': float(self.slider_dd_fade.value()) / 100.0,
            'dd_smooth': self.chk_dd_smooth.isChecked(),
            'dd_mode': self.combo_dd_mode.currentText(),
            'dd_hires': self.chk_dd_hires.isChecked(), 

            'hires_upscaler': self.combo_hires_upscaler.currentText(),
            
            # Sliders are handled below
            'bmab_enabled': self.chk_bmab_enabled.isChecked(), 

            # [New] Composition
            'resize_enable': self.chk_resize_enable.isChecked(),
            'resize_align': self.combo_resize_align.currentText(),
            'bmab_landscape_detail': self.chk_landscape_detail.isChecked(),
            
            # [New] BMAB Edge
            'bmab_edge_enabled': self.g_edge.isChecked(),

            # [New] Sort Method
            'sort_method': "신뢰도" if self.radio_sort_conf.isChecked() else
                           "위치(좌에서 우)" if self.radio_sort_lr.isChecked() else
                           "위치(우에서 좌)" if self.radio_sort_rl.isChecked() else
                           "위치 (중앙에서 바깥)" if self.radio_sort_center.isChecked() else
                           "영역 (대형에서 소형)" if self.radio_sort_area.isChecked() else
                           "위치(위에서 아래)" if self.radio_sort_tb.isChecked() else
                           "위치(아래에서 위)" if self.radio_sort_bt.isChecked() else "신뢰도",

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
            'use_hires_fix': self.chk_hires.isChecked(),
            'use_hires_fix': self.chk_hires.isChecked(),
            'sep_noise': self.chk_sep_noise.isChecked(),
            
            # [New] Detail Daemon Params
            'dd_enabled': self.chk_dd_active.isChecked(),
            'dd_start': self.slider_dd_start.value() / 100.0,
            'dd_end': self.slider_dd_end.value() / 100.0,
            'dd_amount': self.slider_dd_amount.value() / 100.0,
            'dd_bias': self.slider_dd_bias.value() / 100.0,
            'dd_exponent': self.slider_dd_exponent.value() / 100.0,
            'dd_start_offset': self.slider_dd_start_offset.value() / 100.0,
            'dd_end_offset': self.slider_dd_end_offset.value() / 100.0,
            'dd_fade': self.slider_dd_fade.value() / 100.0,
            'dd_smooth': self.chk_dd_smooth.isChecked(),
            'dd_mode': self.combo_dd_mode.currentText(),
            'post_sharpen': self.settings['post_sharpen']['widget'].value(),
            'restore_face_strength': self.settings['restore_face_strength']['widget'].value(),
        }

        # 슬라이더 값들 병합 (Max Face Ratio 등 포함)
        for key, data in self.settings.items():
            widget = data['widget']
            val = widget.value()
            # [Fix] % 단위 UI 값을 Ratio(0~1)로 변환하여 로직에 전달 (Face Ratio 특수 처리)
            if key in ['min_face_ratio', 'max_face_ratio']:
                val /= 100.0
            cfg[key] = val
            
        cfg['seed'] = -1
        return cfg