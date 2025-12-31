import sys
import os
from core.config import config_instance as cfg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QCheckBox, QTextEdit, QGroupBox, 
                             QFormLayout, QDoubleSpinBox, QSlider, QTabWidget,
                             QScrollArea, QSpinBox, QRadioButton, QButtonGroup, QGridLayout)
from PyQt6.QtCore import Qt

class AdetailerUnitWidget(QWidget):
    def __init__(self, unit_name="패스 1"):
        super().__init__()
        self.unit_name = unit_name
        self.settings = {}  # Store references to input widgets
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content_widget = QWidget()
        self.layout = QGridLayout(content_widget)
        self.layout.setSpacing(15)

        # =================================================
        # 1. 모델 및 모드 설정 (Model & Mode)
        # =================================================
        group_model = QGroupBox("모델 및 모드 설정")
        layout_model = QVBoxLayout()
        
        # Enable Checkbox
        self.chk_enable = QCheckBox(f"이 탭 활성화 (Enable {self.unit_name})")
        self.chk_enable.setChecked(True if "1" in self.unit_name else False)
        
        # Detection Method (Radio)
        layout_radio = QHBoxLayout()
        layout_radio.addWidget(QLabel("감지 방식:"))
        self.radio_yolo = QRadioButton("YOLO (객체 감지)")
        self.radio_sam = QRadioButton("SAM3 (세그먼트)")
        self.radio_yolo.setChecked(True)
        btn_group_method = QButtonGroup(self)
        btn_group_method.addButton(self.radio_yolo)
        btn_group_method.addButton(self.radio_sam)
        layout_radio.addWidget(self.radio_yolo)
        layout_radio.addWidget(self.radio_sam)
        layout_radio.addStretch()
        
        # Model Dropdown
        self.combo_model = QComboBox()
        # Populate from config path
        sam_dir = cfg.get_path('sam')
        if sam_dir and os.path.exists(sam_dir):
            models = [f for f in os.listdir(sam_dir) if f.endswith('.pt') or f.endswith('.pth')]
            self.combo_model.addItems(models)
        else:
            self.combo_model.addItems(["face_yolov8n.pt", "person_yolov8n-seg.pt", "hand_yolov8n.pt"])
        
        layout_model.addWidget(self.chk_enable)
        layout_model.addLayout(layout_radio)
        layout_model.addWidget(QLabel("YOLO 모델:"))
        layout_model.addWidget(self.combo_model)
        group_model.setLayout(layout_model)
        self.layout.addWidget(group_model, 0, 0)

        # =================================================
        # 2. 인페인팅 프롬프트 (Prompts)
        # =================================================
        group_prompt = QGroupBox("인페인팅 프롬프트")
        layout_prompt = QVBoxLayout()
        
        self.txt_pos = QTextEdit()
        self.txt_pos.setPlaceholderText("Positive Prompt (e.g., detailed face, high quality...)")
        self.txt_pos.setMaximumHeight(60)
        
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlaceholderText("Negative Prompt (e.g., low quality, blurry...)")
        self.txt_neg.setMaximumHeight(45)
        
        layout_prompt.addWidget(self.txt_pos)
        layout_prompt.addWidget(self.txt_neg)
        group_prompt.setLayout(layout_prompt)
        self.layout.addWidget(group_prompt, 1, 0)

        # =================================================
        # 3. 감지 설정 (Detection)
        # =================================================
        group_detect = QGroupBox("감지 설정 (Detection)")
        layout_detect = QGridLayout()
        
        # Thresholds
        self.add_slider_row(layout_detect, 0, "신뢰도(Conf):", "conf", 0.0, 1.0, 0.35, 0.01)
        self.add_slider_row(layout_detect, 1, "마스크 최소 비율:", "min_area", 0.0, 1.0, 0.0, 0.01)
        self.add_slider_row(layout_detect, 2, "마스크 최대 비율:", "max_area", 0.0, 1.0, 1.0, 0.01)
        
        # Filtering Criteria
        layout_filter = QHBoxLayout()
        layout_filter.addWidget(QLabel("필터링 기준:"))
        self.radio_area = QRadioButton("면적")
        self.radio_conf = QRadioButton("신뢰도")
        self.radio_conf.setChecked(True)
        btn_group_filter = QButtonGroup(self)
        btn_group_filter.addButton(self.radio_area)
        btn_group_filter.addButton(self.radio_conf)
        layout_filter.addWidget(self.radio_area)
        layout_filter.addWidget(self.radio_conf)
        
        # Top K
        spin_top_k = QSpinBox()
        spin_top_k.setValue(0)
        
        layout_detect.addLayout(layout_filter, 3, 0, 1, 2)
        layout_detect.addWidget(QLabel("상위 K개만 사용 (0=전체):"), 4, 0)
        layout_detect.addWidget(spin_top_k, 4, 1)
        
        group_detect.setLayout(layout_detect)
        self.layout.addWidget(group_detect, 2, 0)

        # =================================================
        # 4. 마스크 전처리 (Mask Preprocessing)
        # =================================================
        group_mask = QGroupBox("마스크 전처리")
        layout_mask = QGridLayout()
        
        self.add_slider_row(layout_mask, 0, "X축 오프셋:", "x_offset", -200, 200, 0, 1)
        self.add_slider_row(layout_mask, 1, "Y축 오프셋:", "y_offset", -200, 200, 0, 1)
        self.add_slider_row(layout_mask, 2, "침식(-)/확장(+):", "dilation", -64, 64, 4, 1)
        
        layout_merge = QHBoxLayout()
        layout_merge.addWidget(QLabel("마스크 병합 모드:"))
        self.radio_merge_none = QRadioButton("없음")
        self.radio_merge_merge = QRadioButton("병합")
        self.radio_merge_invert = QRadioButton("병합 후 반전")
        self.radio_merge_merge.setChecked(True)
        layout_merge.addWidget(self.radio_merge_none)
        layout_merge.addWidget(self.radio_merge_merge)
        layout_merge.addWidget(self.radio_merge_invert)
        
        layout_mask.addLayout(layout_merge, 3, 0, 1, 3)
        group_mask.setLayout(layout_mask)
        self.layout.addWidget(group_mask, 3, 0)

        # =================================================
        # 5. 인페인팅 (Inpainting)
        # =================================================
        group_inpaint = QGroupBox("인페인팅 (Inpainting)")
        layout_inpaint = QGridLayout()
        
        self.add_slider_row(layout_inpaint, 0, "마스크 블러:", "blur", 0, 64, 4, 1)
        self.add_slider_row(layout_inpaint, 1, "디노이징 강도:", "denoise", 0.0, 1.0, 0.4, 0.01)
        
        # Inpaint Area
        layout_area = QHBoxLayout()
        self.chk_inpaint_mask_only = QCheckBox("마스크 영역만 인페인팅")
        self.chk_inpaint_mask_only.setChecked(True)
        self.chk_use_sep_res = QCheckBox("별도 해상도 사용")
        layout_area.addWidget(self.chk_inpaint_mask_only)
        layout_area.addWidget(self.chk_use_sep_res)
        
        self.add_slider_row(layout_inpaint, 2, "패딩(px):", "padding", 0, 256, 32, 1)
        
        layout_inpaint.addLayout(layout_area, 3, 0, 1, 3)
        
        # Resolution Sliders
        self.add_slider_row(layout_inpaint, 4, "너비:", "inpaint_width", 64, 2048, 512, 8)
        self.add_slider_row(layout_inpaint, 5, "높이:", "inpaint_height", 64, 2048, 512, 8)

        group_inpaint.setLayout(layout_inpaint)
        self.layout.addWidget(group_inpaint, 0, 1)

        # =================================================
        # 6. 고급 모델 설정 (Advanced)
        # =================================================
        group_adv = QGroupBox("고급 모델 설정")
        layout_adv = QGridLayout()
        
        # Checkboxes and Sliders mixed
        self.chk_sep_steps = QCheckBox("별도 단계 사용")
        layout_adv.addWidget(self.chk_sep_steps, 0, 0)
        self.add_slider_row(layout_adv, 0, "단계(Steps):", "steps", 1, 150, 20, 1, start_col=1)
        
        self.chk_sep_cfg = QCheckBox("별도 CFG 사용")
        layout_adv.addWidget(self.chk_sep_cfg, 1, 0)
        self.add_slider_row(layout_adv, 1, "CFG 스케일:", "cfg_scale", 1.0, 30.0, 7.0, 0.5, start_col=1)
        
        self.chk_sep_ckpt = QCheckBox("별도 체크포인트 사용")
        self.combo_sep_ckpt = QComboBox()
        self.combo_sep_ckpt.addItem("Use Global")
        ckpt_dir = cfg.get_path('checkpoint')
        if ckpt_dir and os.path.exists(ckpt_dir):
            self.combo_sep_ckpt.addItems([f for f in os.listdir(ckpt_dir) if f.endswith(('.ckpt', '.safetensors'))])
            
        layout_adv.addWidget(self.chk_sep_ckpt, 2, 0)
        layout_adv.addWidget(self.combo_sep_ckpt, 2, 1, 1, 2)
        
        self.chk_sep_vae = QCheckBox("별도 VAE 사용")
        self.combo_sep_vae = QComboBox()
        self.combo_sep_vae.addItem("Use Global")
        vae_dir = cfg.get_path('vae')
        if vae_dir and os.path.exists(vae_dir):
            self.combo_sep_vae.addItems([f for f in os.listdir(vae_dir) if f.endswith(('.pt', '.ckpt', '.safetensors'))])

        layout_adv.addWidget(self.chk_sep_vae, 3, 0)
        layout_adv.addWidget(self.combo_sep_vae, 3, 1, 1, 2)

        self.chk_sep_sampler = QCheckBox("별도 샘플러 사용")
        self.combo_sep_sampler = QComboBox()
        self.combo_sep_sampler.addItems(["Euler a", "DPM++ 2M"])
        self.combo_sep_scheduler = QComboBox()
        self.combo_sep_scheduler.addItems(["Karras", "Exponential"])
        layout_adv.addWidget(self.chk_sep_sampler, 4, 0)
        layout_adv.addWidget(self.combo_sep_sampler, 4, 1)
        layout_adv.addWidget(self.combo_sep_scheduler, 4, 2)

        # Noise Multiplier / Clip Skip
        self.chk_sep_noise = QCheckBox("별도 노이즈 사용")
        layout_adv.addWidget(self.chk_sep_noise, 5, 0)
        self.add_slider_row(layout_adv, 5, "노이즈 배율:", "noise_mult", 0.5, 1.5, 1.0, 0.05, start_col=1)

        self.chk_sep_clip = QCheckBox("별도 클립 건너뛰기")
        layout_adv.addWidget(self.chk_sep_clip, 6, 0)
        self.add_slider_row(layout_adv, 6, "클립 건너뛰기:", "clip_skip", 1, 12, 1, 1, start_col=1)
        
        self.chk_restore_face = QCheckBox("작업 후 얼굴 보정 (Restore Face)")
        layout_adv.addWidget(self.chk_restore_face, 7, 0, 1, 3)

        group_adv.setLayout(layout_adv)
        self.layout.addWidget(group_adv, 1, 1)

        # =================================================
        # 7. 컨트롤넷 (ControlNet)
        # =================================================
        group_cn = QGroupBox("컨트롤넷 (ControlNet)")
        layout_cn = QGridLayout()
        
        layout_cn.addWidget(QLabel("모델:"), 0, 0)
        self.combo_cn_model = QComboBox()
        self.combo_cn_model.addItem("None")
        cn_dir = cfg.get_path('controlnet')
        if cn_dir and os.path.exists(cn_dir):
            self.combo_cn_model.addItems([f for f in os.listdir(cn_dir) if f.endswith(('.pth', '.safetensors'))])
        layout_cn.addWidget(self.combo_cn_model, 0, 1, 1, 2)
        
        self.add_slider_row(layout_cn, 1, "가중치:", "cn_weight", 0.0, 2.0, 1.0, 0.05)
        self.add_slider_row(layout_cn, 2, "가이던스 시작:", "cn_start", 0.0, 1.0, 0.0, 0.01)
        self.add_slider_row(layout_cn, 3, "가이던스 끝:", "cn_end", 0.0, 1.0, 1.0, 0.01)
        
        group_cn.setLayout(layout_cn)
        self.layout.addWidget(group_cn, 4, 0)

        self.layout.setRowStretch(5, 1)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        
        scroll.setWidget(content_widget)
        
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def add_slider_row(self, layout, row, label_text, key, min_val, max_val, default_val, step, start_col=0):
        """Helper to create Label | Slider | SpinBox row"""
        label = QLabel(label_text)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        # Slider works with ints, so scale floats if needed
        is_float = isinstance(default_val, float)
        scale = 100 if is_float else 1
        
        slider.setRange(int(min_val * scale), int(max_val * scale))
        slider.setValue(int(default_val * scale))
        
        if is_float:
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
        else:
            spin = QSpinBox()
            
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        
        # Connect signals
        slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * scale)))
        
        layout.addWidget(label, row, start_col)
        layout.addWidget(slider, row, start_col + 1)
        layout.addWidget(spin, row, start_col + 2)
        
        # Store reference for get_config
        self.settings[key] = spin

    def get_config(self):
        """UI 요소에서 설정값을 읽어 딕셔너리로 반환"""
        cfg = {
            'enabled': self.chk_enable.isChecked(),
            'model': self.combo_model.currentText(),
            'use_sam': self.radio_sam.isChecked(),
            'pos_prompt': self.txt_pos.toPlainText(),
            'neg_prompt': self.txt_neg.toPlainText(),
            'merge_mode': "Merge" if self.radio_merge_merge.isChecked() else ("Merge and Invert" if self.radio_merge_invert.isChecked() else "None"),
            
            # Advanced Checkboxes
            'sep_steps': self.chk_sep_steps.isChecked(),
            'sep_cfg': self.chk_sep_cfg.isChecked(),
            'sep_ckpt': self.chk_sep_ckpt.isChecked(),
            'sep_ckpt_name': self.combo_sep_ckpt.currentText(),
            'sep_vae': self.chk_sep_vae.isChecked(),
            'sep_vae_name': self.combo_sep_vae.currentText(),
            'sep_sampler': self.chk_sep_sampler.isChecked(),
            'sampler': f"{self.combo_sep_sampler.currentText()} {self.combo_sep_scheduler.currentText()}",
            'sep_noise': self.chk_sep_noise.isChecked(),
            'sep_clip': self.chk_sep_clip.isChecked(),
            'restore_face': self.chk_restore_face.isChecked(),
            
            # ControlNet
            'use_controlnet': self.combo_cn_model.currentText() != "None",
            'cn_model': self.combo_cn_model.currentText(),
            
            # Inpaint Area
            'inpaint_mask_only': self.chk_inpaint_mask_only.isChecked(),
            'use_sep_res': self.chk_use_sep_res.isChecked(),
        }

        # Collect values from sliders/spinboxes
        for key, widget in self.settings.items():
            cfg[key] = widget.value()
            
        # Default seed (not in UI yet, assume random)
        cfg['seed'] = -1
        
        return cfg