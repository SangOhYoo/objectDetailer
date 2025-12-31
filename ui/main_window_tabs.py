import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QCheckBox, QTextEdit, QGroupBox, 
                             QFormLayout, QDoubleSpinBox, QSlider, QTabWidget,
                             QScrollArea)
from PyQt6.QtCore import Qt

class AdetailerUnitWidget(QWidget):
    """
    Bing-su/adetailer의 단일 Unit UI를 이식하고 BMAB/SAM 기능을 확장한 클래스.
    설정값(Config)을 생성하여 Worker에게 전달하는 역할을 수행함.
    """
    def __init__(self, unit_name="1st"):
        super().__init__()
        self.unit_name = unit_name
        self.init_ui()

    def init_ui(self):
        # 1. Scroll Area (옵션이 많으므로 스크롤 필수)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        self.layout = QVBoxLayout(content_widget)

        # --- A. Header (Enable & Detection Model) ---
        header_group = QGroupBox(f"ADetailer Unit: {self.unit_name}")
        header_layout = QFormLayout()

        self.chk_enable = QCheckBox("Enable ADetailer")
        self.chk_enable.setChecked(True if self.unit_name == "1st" else False)

        self.combo_model = QComboBox()
        # 기본 지원 모델 리스트
        models = [
            "face_yolov8n.pt", "face_yolov8s.pt", 
            "hand_yolov8n.pt", "person_yolov8n-seg.pt",
            "mediapipe_face_full", "mediapipe_face_mesh"
        ]
        self.combo_model.addItems(models)

        header_layout.addRow(self.chk_enable)
        header_layout.addRow("ADetailer Model:", self.combo_model)
        header_group.setLayout(header_layout)

        # --- B. Prompts ---
        prompt_group = QGroupBox("Prompts")
        prompt_layout = QVBoxLayout()
        
        self.txt_pos = QTextEdit()
        self.txt_pos.setPlaceholderText("Positive prompt (Optional, e.g., highly detailed face)")
        self.txt_pos.setMaximumHeight(60)
        
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlaceholderText("Negative prompt (Optional, e.g., ugly, deformed)")
        self.txt_neg.setMaximumHeight(45)

        prompt_layout.addWidget(QLabel("Positive:"))
        prompt_layout.addWidget(self.txt_pos)
        prompt_layout.addWidget(QLabel("Negative:"))
        prompt_layout.addWidget(self.txt_neg)
        prompt_group.setLayout(prompt_layout)

        # --- C. BMAB Features (ControlNet & LoRA) ---
        # 형태 보정(Anatomy)을 위한 핵심 기능
        bmab_group = QGroupBox("BMAB / Anatomy Fix")
        bmab_layout = QFormLayout()

        self.chk_controlnet = QCheckBox("Enable ControlNet (Canny)")
        self.chk_controlnet.setToolTip("활성화 시 Inpainting 단계에서 Canny Edge를 추출하여 형태 붕괴를 막습니다.")
        
        self.spin_cn_weight = self._create_spin(0.0, 2.0, 1.0, 0.1)
        
        # Pass 전용 LoRA (얼굴 전용, 손 전용 등)
        self.combo_lora = QComboBox()
        self.combo_lora.addItems(["None", "polyhedron_skin_v1.safetensors", "hand_fixed_v2.safetensors"])
        self.spin_lora_scale = self._create_spin(0.0, 1.0, 0.6, 0.05)

        bmab_layout.addRow(self.chk_controlnet)
        bmab_layout.addRow("CN Weight:", self.spin_cn_weight)
        bmab_layout.addRow("Inject LoRA:", self.combo_lora)
        bmab_layout.addRow("LoRA Scale:", self.spin_lora_scale)
        bmab_group.setLayout(bmab_layout)

        # --- D. Detailed Settings (Tabs) ---
        settings_tabs = QTabWidget()

        # Tab 1: Detection (SAM Option Added)
        tab_detect = QWidget()
        detect_layout = QFormLayout(tab_detect)
        self.spin_conf = self._create_spin(0.0, 1.0, 0.3, 0.05)
        self.chk_use_sam = QCheckBox("Use SAM (Auto-Masking)")
        self.chk_use_sam.setToolTip("활성화 시 Box 대신 SAM을 사용하여 정밀한 누끼 마스크를 생성합니다.")
        
        self.spin_min_area = self._create_spin(0.0, 1.0, 0.0, 0.01)
        self.spin_max_area = self._create_spin(0.0, 1.0, 1.0, 0.01)
        
        detect_layout.addRow("Detection confidence:", self.spin_conf)
        detect_layout.addRow("Use Segment Anything:", self.chk_use_sam)
        detect_layout.addRow("Mask min area ratio:", self.spin_min_area)
        detect_layout.addRow("Mask max area ratio:", self.spin_max_area)
        tab_detect.setLayout(detect_layout)

        # Tab 2: Mask Preprocessing
        tab_mask = QWidget()
        mask_layout = QFormLayout(tab_mask)
        self.spin_x_offset = self._create_spin(-200, 200, 0, 1)
        self.spin_y_offset = self._create_spin(-200, 200, 0, 1)
        self.spin_dilation = self._create_spin(-128, 128, 4, 1)
        self.combo_merge = QComboBox()
        self.combo_merge.addItems(["None", "Merge", "Merge and Invert"])
        mask_layout.addRow("Mask x(→) offset:", self.spin_x_offset)
        mask_layout.addRow("Mask y(↓) offset:", self.spin_y_offset)
        mask_layout.addRow("Erosion(-) / Dilation(+):", self.spin_dilation)
        mask_layout.addRow("Mask merge mode:", self.combo_merge)
        tab_mask.setLayout(mask_layout)

        # Tab 3: Inpainting (Dynamic Denoise is implicit/auto in worker)
        tab_inpaint = QWidget()
        inpaint_layout = QFormLayout(tab_inpaint)
        self.spin_blur = self._create_spin(0, 64, 4, 1)
        self.spin_denoise = self._create_spin(0.0, 1.0, 0.4, 0.01)
        self.spin_padding = self._create_spin(0, 256, 32, 1)
        self.combo_inpaint_area = QComboBox()
        self.combo_inpaint_area.addItems(["Whole picture", "Masked only"])
        self.combo_inpaint_area.setCurrentIndex(1)
        
        inpaint_layout.addRow("Mask blur:", self.spin_blur)
        inpaint_layout.addRow("Base Denoising strength:", self.spin_denoise)
        inpaint_layout.addRow("Inpaint padding:", self.spin_padding)
        inpaint_layout.addRow("Inpaint area:", self.combo_inpaint_area)
        tab_inpaint.setLayout(inpaint_layout)

        settings_tabs.addTab(tab_detect, "Detection")
        settings_tabs.addTab(tab_mask, "Mask Prep")
        settings_tabs.addTab(tab_inpaint, "Inpainting")

        # --- Assemble Layout ---
        self.layout.addWidget(header_group)
        self.layout.addWidget(prompt_group)
        self.layout.addWidget(bmab_group)
        self.layout.addWidget(settings_tabs)
        self.layout.addStretch()

        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0,0,0,0)
        outer_layout.addWidget(scroll)

    def _create_spin(self, min_val, max_val, default_val, step):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        if isinstance(default_val, int) and isinstance(step, int):
            spin.setDecimals(0)
        else:
            spin.setDecimals(2)
        return spin

    def get_config(self):
        """Worker에 전달할 설정 딕셔너리 반환"""
        return {
            "enabled": self.chk_enable.isChecked(),
            "model": self.combo_model.currentText(),
            "pos_prompt": self.txt_pos.toPlainText(),
            "neg_prompt": self.txt_neg.toPlainText(),
            # BMAB
            "use_controlnet": self.chk_controlnet.isChecked(),
            "cn_weight": self.spin_cn_weight.value(),
            "lora_model": self.combo_lora.currentText(),
            "lora_scale": self.spin_lora_scale.value(),
            # Detection & SAM
            "conf": self.spin_conf.value(),
            "use_sam": self.chk_use_sam.isChecked(),
            "min_area": self.spin_min_area.value(),
            "max_area": self.spin_max_area.value(),
            # Mask
            "x_offset": int(self.spin_x_offset.value()),
            "y_offset": int(self.spin_y_offset.value()),
            "dilation": int(self.spin_dilation.value()),
            "merge_mode": self.combo_merge.currentText(),
            # Inpaint
            "blur": int(self.spin_blur.value()),
            "denoise": self.spin_denoise.value(),
            "padding": int(self.spin_padding.value()),
            "inpaint_mode": self.combo_inpaint_area.currentText()
        }