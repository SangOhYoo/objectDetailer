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
        # 스크롤 영역 설정
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        content_widget = QWidget()
        # 전체 레이아웃: 좌우 2열 그리드
        self.layout = QGridLayout(content_widget)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # =================================================
        # [LEFT COLUMN] 1. 모델 및 모드 설정
        # =================================================
        group_model = QGroupBox("1. 모델 및 모드 (Model & Mode)")
        layout_model = QGridLayout()
        layout_model.setContentsMargins(5, 8, 5, 5)
        
        self.chk_enable = QCheckBox(f"탭 활성화 ({self.unit_name})")
        self.chk_enable.setStyleSheet("font-weight: bold; color: #3498db;")
        is_enabled = self.saved_config.get('enabled', ("1" in self.unit_name))
        self.chk_enable.setChecked(is_enabled)
        
        self.radio_yolo = QRadioButton("YOLO (객체)")
        self.radio_sam = QRadioButton("SAM3 (세그먼트)")
        
        if self.saved_config.get('use_sam', False):
            self.radio_sam.setChecked(True)
        else:
            self.radio_yolo.setChecked(True)

        btn_group = QButtonGroup(self)
        btn_group.addButton(self.radio_yolo)
        btn_group.addButton(self.radio_sam)
        
        self.combo_model = QComboBox()
        sam_dir = cfg.get_path('sam')
        if sam_dir and os.path.exists(sam_dir):
            models = [f for f in os.listdir(sam_dir) if f.endswith('.pt') or f.endswith('.pth')]
            self.combo_model.addItems(models)
        else:
            self.combo_model.addItems(["face_yolov8n.pt", "person_yolov8n-seg.pt", "hand_yolov8n.pt"])
        
        # 모델명 복원
        saved_model = self.saved_config.get('detector_model', '')
        if saved_model:
            index = self.combo_model.findText(saved_model)
            if index >= 0: self.combo_model.setCurrentIndex(index)
            
        layout_model.addWidget(self.chk_enable, 0, 0, 1, 2)
        layout_model.addWidget(QLabel("방식:"), 1, 0)
        layout_model.addWidget(self.radio_yolo, 1, 1)
        layout_model.addWidget(self.radio_sam, 1, 2)
        layout_model.addWidget(QLabel("모델:"), 2, 0)
        layout_model.addWidget(self.combo_model, 2, 1, 1, 2)
        
        group_model.setLayout(layout_model)
        group_model.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.layout.addWidget(group_model, 0, 0)

        # =================================================
        # [LEFT] 3. 감지 및 필터 (Detection) - Max Size 추가됨
        # =================================================
        group_detect = QGroupBox("3. 감지 및 필터 (Detection)")
        layout_detect = QGridLayout()
        layout_detect.setContentsMargins(5, 8, 5, 5)
        
        self.combo_gender = QComboBox()
        self.combo_gender.addItems(["All (성별무관)", "Male", "Female"])
        saved_gender = self.saved_config.get('gender_filter', "All")
        if saved_gender == "All":
            self.combo_gender.setCurrentIndex(0)
        else:
            self.combo_gender.setCurrentText(saved_gender)

        self.chk_ignore_edge = QCheckBox("Edge 무시")
        self.chk_ignore_edge.setChecked(self.saved_config.get('ignore_edge_touching', False))
        
        self.chk_anatomy = QCheckBox("해부학 검증")
        self.chk_anatomy.setChecked(self.saved_config.get('anatomy_check', True))
        
        layout_top_detect = QHBoxLayout()
        layout_top_detect.setContentsMargins(0, 0, 0, 0)
        layout_top_detect.addWidget(QLabel("성별:"))
        layout_top_detect.addWidget(self.combo_gender)
        layout_top_detect.addWidget(self.chk_ignore_edge)
        layout_top_detect.addWidget(self.chk_anatomy)
        layout_detect.addLayout(layout_top_detect, 0, 0, 1, 3)

        self.add_slider_row(layout_detect, 1, "신뢰도(Conf):", "conf_thresh", 0.0, 1.0, 0.35, 0.01)
        # [복구] 최소 크기와 최대 크기를 나란히 배치
        self.add_slider_row(layout_detect, 2, "최소 크기(%):", "min_face_ratio", 0.0, 1.0, 0.01, 0.01)
        self.add_slider_row(layout_detect, 3, "최대 크기(%):", "max_face_ratio", 0.0, 1.0, 1.00, 0.01) # [누락 복구]
        
        layout_detect.addWidget(QLabel("최대 검출 수:"), 4, 0)
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setValue(self.saved_config.get('max_det', 20))
        layout_detect.addWidget(self.spin_top_k, 4, 1)

        group_detect.setLayout(layout_detect)
        self.layout.addWidget(group_detect, 1, 0)

        # =================================================
        # [LEFT] 5. 인페인팅 설정 (Inpaint) - Mask Merge 추가됨
        # =================================================
        group_inpaint = QGroupBox("5. 인페인팅 설정 (Inpaint)")
        layout_inpaint = QGridLayout()
        layout_inpaint.setContentsMargins(5, 8, 5, 5)
        
        self.add_slider_row(layout_inpaint, 0, "디노이징:", "denoising_strength", 0.0, 1.0, 0.4, 0.01)
        self.add_slider_row(layout_inpaint, 1, "패딩(px):", "crop_padding", 0, 256, 32, 1)
        
        # 인페인팅 해상도
        layout_res = QHBoxLayout()
        layout_res.addWidget(QLabel("강제 해상도:"))
        self.spin_inpaint_w = QSpinBox()
        self.spin_inpaint_w.setRange(0, 2048)
        self.spin_inpaint_w.setValue(self.saved_config.get('inpaint_width', 0))
        self.spin_inpaint_h = QSpinBox()
        self.spin_inpaint_h.setRange(0, 2048)
        self.spin_inpaint_h.setValue(self.saved_config.get('inpaint_height', 0))
        layout_res.addWidget(self.spin_inpaint_w)
        layout_res.addWidget(QLabel("x"))
        layout_res.addWidget(self.spin_inpaint_h)
        layout_inpaint.addLayout(layout_res, 2, 0, 1, 3)

        layout_color = QHBoxLayout()
        layout_color.addWidget(QLabel("색감 보정:"))
        self.combo_color_fix = QComboBox()
        self.combo_color_fix.addItems(["None", "Wavelet", "Adain"])
        self.combo_color_fix.setCurrentText(self.saved_config.get('color_fix', "None"))
        layout_color.addWidget(self.combo_color_fix)
        layout_inpaint.addLayout(layout_color, 3, 0, 1, 3)
        
        # [복구] 마스크 병합 & 노이즈 마스크
        self.chk_mask_merge = QCheckBox("마스크 병합(Merge)") # [누락 복구]
        self.chk_mask_merge.setChecked(self.saved_config.get('mask_merge_mode', False))
        
        self.chk_noise_mask = QCheckBox("노이즈 마스크")
        self.chk_noise_mask.setChecked(self.saved_config.get('use_noise_mask', False))
        
        self.chk_auto_rotate = QCheckBox("🔄 자동 회전 보정 (Auto Rotate)")
        self.chk_auto_rotate.setChecked(self.saved_config.get('auto_rotate', True))
        
        layout_inpaint.addWidget(self.chk_mask_merge, 4, 0)
        layout_inpaint.addWidget(self.chk_noise_mask, 4, 1)
        layout_inpaint.addWidget(self.chk_auto_rotate, 5, 0, 1, 2)

        group_inpaint.setLayout(layout_inpaint)
        self.layout.addWidget(group_inpaint, 2, 0)

        # =================================================
        # [LEFT BOTTOM] 8. SAM 세부 설정 (SAM)
        # =================================================
        group_sam = QGroupBox("8. SAM 세부 설정 (SAM Settings)")
        layout_sam = QGridLayout()
        layout_sam.setContentsMargins(5, 8, 5, 5)
        
        self.add_slider_row(layout_sam, 0, "Points/Side:", "sam_points_per_side", 1, 64, 32, 1)
        self.add_slider_row(layout_sam, 1, "Pred IOU:", "sam_pred_iou_thresh", 0.0, 1.0, 0.88, 0.01)
        self.add_slider_row(layout_sam, 2, "Stability:", "sam_stability_score_thresh", 0.0, 1.0, 0.95, 0.01)
        
        group_sam.setLayout(layout_sam)
        self.layout.addWidget(group_sam, 3, 0)


        # =================================================
        # [RIGHT COLUMN] 2. 프롬프트 및 자동화
        # =================================================
        group_prompt = QGroupBox("2. 프롬프트 및 자동화")
        layout_prompt = QVBoxLayout()
        layout_prompt.setContentsMargins(5, 8, 5, 5)
        
        self.chk_auto_prompt = QCheckBox("✨ 자동 프롬프트 주입 (Auto Injection)")
        self.chk_auto_prompt.setChecked(self.saved_config.get('auto_prompt_injection', True))
        self.chk_auto_prompt.setStyleSheet("color: #e67e22; font-weight: bold;")

        self.txt_pos = QTextEdit()
        self.txt_pos.setPlaceholderText("Positive Prompt (e.g. detailed face, high quality)")
        self.txt_pos.setText(self.saved_config.get('pos_prompt', ""))
        self.txt_pos.setMaximumHeight(50)
        # 50:50 비율 유지를 위해 Expanding 방지
        self.txt_pos.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlaceholderText("Negative Prompt (e.g. low quality, blurry)")
        self.txt_neg.setText(self.saved_config.get('neg_prompt', ""))
        self.txt_neg.setMaximumHeight(40)
        # 50:50 비율 유지를 위해 Expanding 방지
        self.txt_neg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        layout_prompt.addWidget(self.chk_auto_prompt)
        layout_prompt.addWidget(self.txt_pos)
        layout_prompt.addWidget(self.txt_neg)
        group_prompt.setLayout(layout_prompt)
        group_prompt.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.layout.addWidget(group_prompt, 0, 1)

        # =================================================
        # [RIGHT] 4. 마스크 전처리 (Mask)
        # =================================================
        group_mask = QGroupBox("4. 마스크 전처리 (Mask)")
        layout_mask = QGridLayout()
        layout_mask.setContentsMargins(5, 8, 5, 5)
        
        self.add_slider_row(layout_mask, 1, "확장(Dilation):", "mask_dilation", -64, 64, 4, 1)
        self.add_slider_row(layout_mask, 2, "침식(Erosion):", "mask_erosion", 0, 64, 0, 1)
        self.add_slider_row(layout_mask, 3, "블러(Blur):", "mask_blur", 0, 64, 12, 1)
        self.add_slider_row(layout_mask, 4, "X 오프셋:", "x_offset", -100, 100, 0, 1)
        self.add_slider_row(layout_mask, 5, "Y 오프셋:", "y_offset", -100, 100, 0, 1)
        
        group_mask.setLayout(layout_mask)
        self.layout.addWidget(group_mask, 1, 1)

        # =================================================
        # [RIGHT] 6. ControlNet & BMAP - Preprocessor 추가됨
        # =================================================
        group_adv = QGroupBox("6. ControlNet & BMAP")
        layout_adv = QGridLayout()
        layout_adv.setContentsMargins(5, 8, 5, 5)
        
        # 1. Model
        self.combo_cn_model = QComboBox()
        self.combo_cn_model.addItem("None")
        cn_dir = cfg.get_path('controlnet')
        if cn_dir and os.path.exists(cn_dir):
            self.combo_cn_model.addItems([f for f in os.listdir(cn_dir)])
        
        saved_cn = self.saved_config.get('control_model', 'None')
        if not self.saved_config.get('use_controlnet', False): saved_cn = "None"
        idx = self.combo_cn_model.findText(saved_cn)
        if idx >= 0: self.combo_cn_model.setCurrentIndex(idx)
        
        layout_adv.addWidget(QLabel("CN 모델:"), 0, 0)
        layout_adv.addWidget(self.combo_cn_model, 0, 1, 1, 2)

        # 2. Module (Preprocessor) [누락 복구]
        self.combo_cn_module = QComboBox()
        self.combo_cn_module.addItems(["inpaint_global_harmonious", "inpaint_only", "lineart_realistic", "canny", "depth_midas", "None"])
        self.combo_cn_module.setCurrentText(self.saved_config.get('control_module', "inpaint_global_harmonious"))
        
        layout_adv.addWidget(QLabel("전처리(Module):"), 1, 0)
        layout_adv.addWidget(self.combo_cn_module, 1, 1, 1, 2)
        
        self.add_slider_row(layout_adv, 2, "CN 가중치:", "control_weight", 0.0, 2.0, 1.0, 0.1)
        self.add_slider_row(layout_adv, 3, "시작(Start):", "guidance_start", 0.0, 1.0, 0.0, 0.05)
        self.add_slider_row(layout_adv, 4, "종료(End):", "guidance_end", 0.0, 1.0, 1.0, 0.05)
        
        self.chk_hires = QCheckBox("Hires Fix")
        self.chk_hires.setChecked(self.saved_config.get('use_hires_fix', False))
        
        self.chk_sep_noise = QCheckBox("별도 노이즈")
        self.chk_sep_noise.setChecked(self.saved_config.get('sep_noise', False))

        layout_adv.addWidget(self.chk_hires, 5, 0)
        layout_adv.addWidget(self.chk_sep_noise, 5, 1)
        
        self.add_slider_row(layout_adv, 6, "업스케일:", "upscale_factor", 1.0, 2.0, 1.5, 0.1)
        self.add_slider_row(layout_adv, 7, "노이즈 배율:", "noise_multiplier", 0.5, 1.5, 1.0, 0.05)
        
        group_adv.setLayout(layout_adv)
        self.layout.addWidget(group_adv, 2, 1)

        # =================================================
        # [RIGHT BOTTOM] 7. 개별 패스 고급 설정 (Overrides)
        # =================================================
        group_override = QGroupBox("7. 개별 패스 고급 설정 (Overrides)")
        layout_override = QGridLayout()
        layout_override.setContentsMargins(5, 8, 5, 5)

        # (1) Checkpoint & VAE
        self.chk_sep_ckpt = QCheckBox("CKPT")
        self.chk_sep_ckpt.setChecked(self.saved_config.get('sep_ckpt', False))
        self.combo_sep_ckpt = QComboBox()
        self.combo_sep_ckpt.addItem("Use Global")
        ckpt_dir = cfg.get_path('checkpoint')
        if ckpt_dir and os.path.exists(ckpt_dir):
            self.combo_sep_ckpt.addItems([f for f in os.listdir(ckpt_dir) if f.endswith(('.ckpt', '.safetensors'))])
        self.combo_sep_ckpt.setCurrentText(self.saved_config.get('sep_ckpt_name', 'Use Global'))
        
        self.chk_sep_vae = QCheckBox("VAE")
        self.chk_sep_vae.setChecked(self.saved_config.get('sep_vae', False))
        self.combo_sep_vae = QComboBox()
        self.combo_sep_vae.addItem("Use Global")
        vae_dir = cfg.get_path('vae')
        if vae_dir and os.path.exists(vae_dir):
            self.combo_sep_vae.addItems([f for f in os.listdir(vae_dir) if f.endswith(('.pt', '.ckpt', '.safetensors'))])
        self.combo_sep_vae.setCurrentText(self.saved_config.get('sep_vae_name', 'Use Global'))

        layout_override.addWidget(self.chk_sep_ckpt, 0, 0)
        layout_override.addWidget(self.combo_sep_ckpt, 0, 1)
        layout_override.addWidget(self.chk_sep_vae, 0, 2)
        layout_override.addWidget(self.combo_sep_vae, 0, 3)

        # (2) Sampler & Steps & CFG
        self.chk_sep_sampler = QCheckBox("Sampler")
        self.chk_sep_sampler.setChecked(self.saved_config.get('sep_sampler', False))
        self.combo_sep_sampler = QComboBox()
        self.combo_sep_sampler.addItems(["Euler a", "DPM++ 2M", "DPM++ SDE", "DDIM"])
        self.combo_sep_scheduler = QComboBox()
        self.combo_sep_scheduler.addItems(["Karras", "Exponential", "Automatic"])
        
        # Sampler/Scheduler 복원
        saved_sampler_full = self.saved_config.get('sampler_name', "Euler a Automatic")
        schedulers = ["Karras", "Exponential", "Automatic"]
        found_scheduler = "Automatic"
        found_sampler = saved_sampler_full
        for sch in schedulers:
            if saved_sampler_full.endswith(sch):
                found_scheduler = sch
                found_sampler = saved_sampler_full.replace(sch, "").strip()
                break
        self.combo_sep_sampler.setCurrentText(found_sampler)
        self.combo_sep_scheduler.setCurrentText(found_scheduler)

        self.chk_sep_steps = QCheckBox("Steps")
        self.chk_sep_steps.setChecked(self.saved_config.get('sep_steps', False))
        self.spin_sep_steps = QSpinBox()
        self.spin_sep_steps.setRange(1, 150)
        self.spin_sep_steps.setValue(self.saved_config.get('steps', 20))

        self.chk_sep_cfg = QCheckBox("CFG")
        self.chk_sep_cfg.setChecked(self.saved_config.get('sep_cfg', False))
        self.spin_sep_cfg = QDoubleSpinBox()
        self.spin_sep_cfg.setRange(1.0, 30.0)
        self.spin_sep_cfg.setValue(self.saved_config.get('cfg_scale', 7.0))
        
        layout_override.addWidget(self.chk_sep_sampler, 1, 0)
        layout_override.addWidget(self.combo_sep_sampler, 1, 1)
        layout_override.addWidget(self.combo_sep_scheduler, 1, 2)
        
        layout_sub = QHBoxLayout()
        layout_sub.addWidget(self.chk_sep_steps)
        layout_sub.addWidget(self.spin_sep_steps)
        layout_sub.addWidget(self.chk_sep_cfg)
        layout_sub.addWidget(self.spin_sep_cfg)
        layout_override.addLayout(layout_sub, 1, 3)
        
        # (3) Clip Skip & Restore Face
        self.chk_sep_clip = QCheckBox("Clip Skip")
        self.chk_sep_clip.setChecked(self.saved_config.get('sep_clip', False))
        self.spin_clip = QSpinBox()
        self.spin_clip.setRange(1, 12)
        self.spin_clip.setValue(self.saved_config.get('clip_skip', 2))
        
        self.chk_restore_face = QCheckBox("얼굴 보정 (Restore Face)")
        self.chk_restore_face.setChecked(self.saved_config.get('restore_face', False))
        
        layout_override.addWidget(self.chk_sep_clip, 2, 0)
        layout_override.addWidget(self.spin_clip, 2, 1)
        layout_override.addWidget(self.chk_restore_face, 2, 2, 1, 2)

        group_override.setLayout(layout_override)
        self.layout.addWidget(group_override, 3, 1)

        # 레이아웃 균형 (좌우 50:50)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setRowStretch(4, 1) # 하단 여백 확보
        
        scroll.setWidget(content_widget)
        
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

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
        spin.setFixedWidth(60)
        
        spin.setValue(loaded_val)
        slider.setValue(int(loaded_val * scale))
        
        slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * scale)))
        
        layout.addWidget(label, row, start_col)
        layout.addWidget(slider, row, start_col + 1)
        layout.addWidget(spin, row, start_col + 2)
        
        self.settings[key] = spin

    def get_config(self):
        """Configs.py Key 완전 일치 및 누락 기능 복구 완료"""
        cfg = {
            'enabled': self.chk_enable.isChecked(),
            'detector_model': self.combo_model.currentText(),
            'use_sam': self.radio_sam.isChecked(),
            
            'auto_prompt_injection': self.chk_auto_prompt.isChecked(),
            'gender_filter': self.combo_gender.currentText().split()[0],
            'ignore_edge_touching': self.chk_ignore_edge.isChecked(),
            'anatomy_check': self.chk_anatomy.isChecked(),
            'auto_rotate': self.chk_auto_rotate.isChecked(),
            'color_fix': self.combo_color_fix.currentText(),
            'use_hires_fix': self.chk_hires.isChecked(),
            
            'pos_prompt': self.txt_pos.toPlainText(),
            'neg_prompt': self.txt_neg.toPlainText(),
            'max_det': self.spin_top_k.value(),
            
            'use_controlnet': self.combo_cn_model.currentText() != "None",
            'control_model': self.combo_cn_model.currentText(),
            'control_module': self.combo_cn_module.currentText(), # [복구됨]
            'sep_noise': self.chk_sep_noise.isChecked(),
            
            'inpaint_width': self.spin_inpaint_w.value(),
            'inpaint_height': self.spin_inpaint_h.value(),
            'use_noise_mask': self.chk_noise_mask.isChecked(),
            'mask_merge_mode': self.chk_mask_merge.isChecked(), # [복구됨]

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
            cfg[key] = widget.value()
            
        cfg['seed'] = -1
        return cfg