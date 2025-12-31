import sys
import os
from core.config import config_instance as cfg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QCheckBox, QTextEdit, QGroupBox, 
                             QDoubleSpinBox, QSlider, QScrollArea, QSpinBox, 
                             QRadioButton, QButtonGroup, QGridLayout)
from PyQt6.QtCore import Qt

class AdetailerUnitWidget(QWidget):
    def __init__(self, unit_name="패스 1"):
        super().__init__()
        self.unit_name = unit_name
        self.settings = {}  # 위젯 참조 저장
        self.init_ui()

    def init_ui(self):
        # 가로 세로 스크롤 방지를 위해 ScrollArea 속성 조정
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # 스크롤바 정책: 필요할 때만 표시하지만, 레이아웃 최적화로 최대한 안 뜨게 함
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content_widget = QWidget()
        # 전체를 좌우 2열 그리드로 배치
        self.layout = QGridLayout(content_widget)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # =================================================
        # [그룹 1] 모델 및 모드 (좌측 상단)
        # =================================================
        group_model = QGroupBox("1. 모델 및 모드 설정")
        layout_model = QGridLayout()
        
        self.chk_enable = QCheckBox(f"탭 활성화 ({self.unit_name})")
        # 설정 파일에서 값 불러오기 시도, 없으면 이름 기반 기본값
        saved_enable = cfg.get('ui_settings', self.unit_name, 'enabled')
        self.chk_enable.setChecked(saved_enable if saved_enable is not None else ("1" in self.unit_name))
        
        self.radio_yolo = QRadioButton("YOLO (객체)")
        self.radio_sam = QRadioButton("SAM3 (세그먼트)")
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
            
        layout_model.addWidget(self.chk_enable, 0, 0, 1, 2)
        layout_model.addWidget(QLabel("방식:"), 1, 0)
        layout_model.addWidget(self.radio_yolo, 1, 1)
        layout_model.addWidget(self.radio_sam, 1, 2)
        layout_model.addWidget(QLabel("모델:"), 2, 0)
        layout_model.addWidget(self.combo_model, 2, 1, 1, 2)
        
        group_model.setLayout(layout_model)
        self.layout.addWidget(group_model, 0, 0)

        # =================================================
        # [그룹 2] 프롬프트 (우측 상단)
        # =================================================
        group_prompt = QGroupBox("2. 프롬프트 및 자동화")
        layout_prompt = QVBoxLayout()
        
        # [BMAP] 자동 프롬프트 주입
        self.chk_auto_prompt = QCheckBox("✨ 자동 프롬프트 주입 (Auto Injection)")
        self.chk_auto_prompt.setChecked(True)
        self.chk_auto_prompt.setStyleSheet("color: #4dabf7; font-weight: bold;")

        self.txt_pos = QTextEdit()
        self.txt_pos.setPlaceholderText("Positive Prompt...")
        self.txt_pos.setMaximumHeight(50)
        
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlaceholderText("Negative Prompt...")
        self.txt_neg.setMaximumHeight(40)
        
        layout_prompt.addWidget(self.chk_auto_prompt)
        layout_prompt.addWidget(self.txt_pos)
        layout_prompt.addWidget(self.txt_neg)
        group_prompt.setLayout(layout_prompt)
        self.layout.addWidget(group_prompt, 0, 1)

        # =================================================
        # [그룹 3] 감지 및 필터 (좌측 중단)
        # =================================================
        group_detect = QGroupBox("3. 감지 및 필터 (Detection)")
        layout_detect = QGridLayout()
        
        # [BMAP] 성별 필터, 해부학 검증, 엣지 무시
        self.combo_gender = QComboBox()
        self.combo_gender.addItems(["All", "Male", "Female"])
        self.chk_ignore_edge = QCheckBox("Edge 무시")
        self.chk_anatomy = QCheckBox("해부학 검증") # Anatomy Check
        self.chk_anatomy.setChecked(True)
        
        layout_detect.addWidget(QLabel("성별:"), 0, 0)
        layout_detect.addWidget(self.combo_gender, 0, 1)
        layout_detect.addWidget(self.chk_ignore_edge, 0, 2)
        layout_detect.addWidget(self.chk_anatomy, 0, 3)

        self.add_slider_row(layout_detect, 1, "신뢰도(Conf):", "conf", 0.0, 1.0, 0.35, 0.01)
        self.add_slider_row(layout_detect, 2, "최소 크기(%):", "min_face_ratio", 0.0, 0.5, 0.01, 0.01)
        
        layout_detect.addWidget(QLabel("최대 검출 수:"), 3, 0)
        self.spin_top_k = QSpinBox()
        self.spin_top_k.setValue(20)
        layout_detect.addWidget(self.spin_top_k, 3, 1)

        group_detect.setLayout(layout_detect)
        self.layout.addWidget(group_detect, 1, 0)

        # =================================================
        # [그룹 4] 마스크 전처리 (우측 중단)
        # =================================================
        group_mask = QGroupBox("4. 마스크 전처리 (Mask)")
        layout_mask = QGridLayout()
        
        # [BMAP] 자동 회전
        self.chk_auto_rotate = QCheckBox("🔄 자동 회전 보정 (Auto Rotate)")
        self.chk_auto_rotate.setChecked(True)
        self.chk_auto_rotate.setStyleSheet("color: #e67e22; font-weight: bold;")
        layout_mask.addWidget(self.chk_auto_rotate, 0, 0, 1, 3)

        self.add_slider_row(layout_mask, 1, "침식/확장:", "dilation", -64, 64, 4, 1)
        self.add_slider_row(layout_mask, 2, "X 오프셋:", "x_offset", -100, 100, 0, 1)
        self.add_slider_row(layout_mask, 3, "Y 오프셋:", "y_offset", -100, 100, 0, 1)
        
        group_mask.setLayout(layout_mask)
        self.layout.addWidget(group_mask, 1, 1)

        # =================================================
        # [그룹 5] 인페인팅 설정 (좌측 하단)
        # =================================================
        group_inpaint = QGroupBox("5. 인페인팅 설정 (Inpaint)")
        layout_inpaint = QGridLayout()
        
        self.add_slider_row(layout_inpaint, 0, "디노이징:", "denoise", 0.0, 1.0, 0.4, 0.01)
        self.add_slider_row(layout_inpaint, 1, "마스크 블러:", "blur", 0, 64, 12, 1)
        self.add_slider_row(layout_inpaint, 2, "패딩(px):", "padding", 0, 256, 32, 1)
        
        # [BMAP] 색감 보정
        layout_color = QHBoxLayout()
        layout_color.addWidget(QLabel("색감 보정:"))
        self.combo_color_fix = QComboBox()
        self.combo_color_fix.addItems(["None", "Wavelet", "Adain"])
        layout_color.addWidget(self.combo_color_fix)
        layout_inpaint.addLayout(layout_color, 3, 0, 1, 3)

        group_inpaint.setLayout(layout_inpaint)
        self.layout.addWidget(group_inpaint, 2, 0)

        # =================================================
        # [그룹 6] ControlNet & BMAP (우측 하단)
        # =================================================
        group_adv = QGroupBox("6. ControlNet & BMAP")
        layout_adv = QGridLayout()
        
        self.combo_cn_model = QComboBox()
        self.combo_cn_model.addItem("None")
        cn_dir = cfg.get_path('controlnet')
        if cn_dir and os.path.exists(cn_dir):
            self.combo_cn_model.addItems([f for f in os.listdir(cn_dir)])
        
        layout_adv.addWidget(QLabel("CN 모델:"), 0, 0)
        layout_adv.addWidget(self.combo_cn_model, 0, 1, 1, 2)
        
        self.add_slider_row(layout_adv, 1, "CN 가중치:", "cn_weight", 0.0, 2.0, 1.0, 0.1)
        
        # [BMAP] Hires Fix & 별도 노이즈
        self.chk_hires = QCheckBox("Hires Fix")
        self.chk_sep_noise = QCheckBox("별도 노이즈")
        layout_adv.addWidget(self.chk_hires, 2, 0)
        layout_adv.addWidget(self.chk_sep_noise, 2, 1)
        
        # [BMAP] Upscale & Noise Multiplier
        self.add_slider_row(layout_adv, 3, "업스케일:", "upscale_factor", 1.0, 2.0, 1.5, 0.1)
        self.add_slider_row(layout_adv, 4, "노이즈 배율:", "noise_mult", 0.5, 1.5, 1.0, 0.05)
        
        group_adv.setLayout(layout_adv)
        self.layout.addWidget(group_adv, 2, 1)

        # =================================================
        # [그룹 7] 개별 패스 고급 설정 (기존 소스 복구) - 하단 전체 사용
        # =================================================
        group_override = QGroupBox("7. 개별 패스 고급 설정 (Advanced Overrides) - 기존 기능 복구")
        layout_override = QGridLayout()
        layout_override.setContentsMargins(5, 5, 5, 5)

        # (1) Checkpoint & VAE Override
        self.chk_sep_ckpt = QCheckBox("체크포인트 변경")
        self.combo_sep_ckpt = QComboBox()
        self.combo_sep_ckpt.addItem("Use Global")
        ckpt_dir = cfg.get_path('checkpoint')
        if ckpt_dir and os.path.exists(ckpt_dir):
            self.combo_sep_ckpt.addItems([f for f in os.listdir(ckpt_dir) if f.endswith(('.ckpt', '.safetensors'))])
        
        self.chk_sep_vae = QCheckBox("VAE 변경")
        self.combo_sep_vae = QComboBox()
        self.combo_sep_vae.addItem("Use Global")
        vae_dir = cfg.get_path('vae')
        if vae_dir and os.path.exists(vae_dir):
            self.combo_sep_vae.addItems([f for f in os.listdir(vae_dir) if f.endswith(('.pt', '.ckpt', '.safetensors'))])

        layout_override.addWidget(self.chk_sep_ckpt, 0, 0)
        layout_override.addWidget(self.combo_sep_ckpt, 0, 1)
        layout_override.addWidget(self.chk_sep_vae, 0, 2)
        layout_override.addWidget(self.combo_sep_vae, 0, 3)

        # (2) Sampler & Steps & CFG
        self.chk_sep_sampler = QCheckBox("샘플러 변경")
        self.combo_sep_sampler = QComboBox()
        self.combo_sep_sampler.addItems(["Euler a", "DPM++ 2M", "DPM++ SDE", "DDIM"])
        self.combo_sep_scheduler = QComboBox()
        self.combo_sep_scheduler.addItems(["Karras", "Exponential", "Automatic"])

        self.chk_sep_steps = QCheckBox("Steps")
        self.spin_sep_steps = QSpinBox()
        self.spin_sep_steps.setRange(1, 150)
        self.spin_sep_steps.setValue(20)

        self.chk_sep_cfg = QCheckBox("CFG")
        self.spin_sep_cfg = QDoubleSpinBox()
        self.spin_sep_cfg.setRange(1.0, 30.0)
        self.spin_sep_cfg.setValue(7.0)
        
        layout_override.addWidget(self.chk_sep_sampler, 1, 0)
        layout_override.addWidget(self.combo_sep_sampler, 1, 1)
        layout_override.addWidget(self.combo_sep_scheduler, 1, 2)
        
        # Steps/CFG compact
        layout_sub = QHBoxLayout()
        layout_sub.addWidget(self.chk_sep_steps)
        layout_sub.addWidget(self.spin_sep_steps)
        layout_sub.addWidget(self.chk_sep_cfg)
        layout_sub.addWidget(self.spin_sep_cfg)
        layout_override.addLayout(layout_sub, 1, 3)

        # (3) Clip Skip & Restore Face
        self.chk_sep_clip = QCheckBox("Clip Skip")
        self.spin_clip = QSpinBox()
        self.spin_clip.setRange(1, 12)
        self.spin_clip.setValue(2)
        
        self.chk_restore_face = QCheckBox("얼굴 보정(Restore Face)")
        
        layout_override.addWidget(self.chk_sep_clip, 2, 0)
        layout_override.addWidget(self.spin_clip, 2, 1)
        layout_override.addWidget(self.chk_restore_face, 2, 2, 1, 2)

        group_override.setLayout(layout_override)
        # 하단 전체 너비 사용 (0열부터 2칸 차지)
        self.layout.addWidget(group_override, 3, 0, 1, 2)

        # 레이아웃 균형
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        
        scroll.setWidget(content_widget)
        
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def add_slider_row(self, layout, row, label_text, key, min_val, max_val, default_val, step, start_col=0):
        """슬라이더와 스핀박스를 연결하는 헬퍼 함수"""
        label = QLabel(label_text)
        slider = QSlider(Qt.Orientation.Horizontal)
        
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
        spin.setFixedWidth(60) # UI 깨짐 방지
        
        slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * scale)))
        
        layout.addWidget(label, row, start_col)
        layout.addWidget(slider, row, start_col + 1)
        layout.addWidget(spin, row, start_col + 2)
        
        self.settings[key] = spin

    def get_config(self):
        """현재 UI 상태를 딕셔너리로 반환"""
        cfg = {
            'enabled': self.chk_enable.isChecked(),
            'model': self.combo_model.currentText(),
            'use_sam': self.radio_sam.isChecked(),
            
            # --- 기본 기능 및 BMAP 추가 기능 ---
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
            'cn_model': self.combo_cn_model.currentText(),
            'sep_noise': self.chk_sep_noise.isChecked(),

            # --- 고급 오버라이드 (복구됨) ---
            'sep_ckpt': self.chk_sep_ckpt.isChecked(),
            'sep_ckpt_name': self.combo_sep_ckpt.currentText(),
            'sep_vae': self.chk_sep_vae.isChecked(),
            'sep_vae_name': self.combo_sep_vae.currentText(),
            
            'sep_sampler': self.chk_sep_sampler.isChecked(),
            'sampler': f"{self.combo_sep_sampler.currentText()} {self.combo_sep_scheduler.currentText()}",
            
            'sep_steps': self.chk_sep_steps.isChecked(),
            'steps': self.spin_sep_steps.value(),
            
            'sep_cfg': self.chk_sep_cfg.isChecked(),
            'cfg_scale': self.spin_sep_cfg.value(),
            
            'sep_clip': self.chk_sep_clip.isChecked(),
            'clip_skip': self.spin_clip.value(),
            
            'restore_face': self.chk_restore_face.isChecked(),
        }

        # 슬라이더 값들 병합
        for key, widget in self.settings.items():
            cfg[key] = widget.value()
            
        cfg['seed'] = -1
        return cfg