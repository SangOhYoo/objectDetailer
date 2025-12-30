"""
ui/main_window.py
SAM3_FaceDetailer_Ultimate 메인 윈도우
- PyQt6 기반의 다크 테마 GUI
- 설정값(DetailerConfig) 생성 및 듀얼 워커(ProcessWorker) 통제
- 실시간 로그 및 진행률 모니터링
"""

import sys
import os
import queue
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTextEdit, QTabWidget, 
    QCheckBox, QComboBox, QSlider, QSpinBox, 
    QFileDialog, QProgressBar, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QIcon, QAction

# 모듈 연결
from configs import SystemConfig, DetailerConfig
from ui.workers import InitWorker, ProcessWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. 시스템 설정 로드 (기본값)
        self.sys_config = SystemConfig()
        self.workers = []      # 실행 중인 워커 스레드 리스트
        self.task_queue = queue.Queue() # 작업 대기열 (Thread-Safe)
        
        # 2. UI 초기화
        self.setWindowTitle("SAM3 FaceDetailer Ultimate - Dual GPU Factory")
        self.resize(1200, 850)
        self.setup_ui()
        self.apply_dark_theme()
        
        # 3. 시스템 초기화 워커 실행 (GPU 점검)
        self.init_worker = InitWorker(self.sys_config)
        self.init_worker.log_msg.connect(self.log)
        self.init_worker.finished.connect(lambda: self.log("[System] 초기화 완료. 작업을 시작할 수 있습니다."))
        self.init_worker.start()

    def setup_ui(self):
        """전체 레이아웃 구성"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (좌: 설정 패널 / 우: 로그 및 상태)
        main_layout = QHBoxLayout(central_widget)
        
        # --- [좌측 패널: 설정 (Settings)] ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(500) # 고정 너비
        
        # 탭 위젯 (Main / Detection / Advanced)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_main_tab(), "Main Processing")
        self.tabs.addTab(self.create_detect_tab(), "Detection & Geometry")
        self.tabs.addTab(self.create_advanced_tab(), "Advanced & Control")
        
        left_layout.addWidget(self.tabs)
        
        # 실행 컨트롤 그룹 (하단)
        control_group = QGroupBox("Execution Control")
        control_layout = QVBoxLayout()
        
        # 경로 선택
        path_layout = QHBoxLayout()
        self.btn_input = QPushButton("Input Folder...")
        self.btn_input.clicked.connect(self.select_input_folder)
        self.lbl_input = QLabel("선택된 폴더 없음")
        self.lbl_input.setStyleSheet("color: #aaa; font-style: italic;")
        path_layout.addWidget(self.btn_input)
        path_layout.addWidget(self.lbl_input)
        
        # 시작/중지 버튼
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 START BATCH")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet("background-color: #2e7d32; font-weight: bold; font-size: 14px;")
        self.btn_start.clicked.connect(self.start_processing)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setFixedHeight(50)
        self.btn_stop.setStyleSheet("background-color: #c62828; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        
        control_layout.addLayout(path_layout)
        control_layout.addLayout(btn_layout)
        control_group.setLayout(control_layout)
        
        left_layout.addWidget(control_group)
        
        # --- [우측 패널: 로그 및 뷰어] ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 로그 창
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color: #1e1e1e; color: #00e676; font-family: Consolas;")
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")
        
        right_layout.addWidget(QLabel("Process Log"))
        right_layout.addWidget(self.log_view)
        right_layout.addWidget(self.progress_bar)
        
        # 패널 배치
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])
        
        main_layout.addWidget(splitter)

    # =========================================================
    # 탭 UI 구성 메서드
    # =========================================================
    def create_main_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 모델 선택
        layout.addWidget(QLabel("Checkpoint Model (.safetensors)"))
        self.combo_ckpt = QComboBox()
        self.combo_ckpt.addItems([
            "juggernaut_xl_v9.safetensors", 
            "realisticVisionV60B1_v51VAE.safetensors",
            "dreamshaper_8.safetensors"
        ])
        self.combo_ckpt.setEditable(True) # 직접 입력 가능
        layout.addWidget(self.combo_ckpt)
        
        # 프롬프트
        layout.addWidget(QLabel("Positive Prompt (Auto-injects Gender/Object)"))
        self.txt_pos = QTextEdit()
        self.txt_pos.setPlaceholderText("best quality, detailed face, ...")
        self.txt_pos.setPlainText("best quality, detailed face, high resolution, realistic skin texture")
        self.txt_pos.setMaximumHeight(100)
        layout.addWidget(self.txt_pos)
        
        layout.addWidget(QLabel("Negative Prompt"))
        self.txt_neg = QTextEdit()
        self.txt_neg.setPlainText("(lowres, low quality:1.2), bad anatomy, bad hands, text, watermark")
        self.txt_neg.setMaximumHeight(60)
        layout.addWidget(self.txt_neg)
        
        # Denoising Strength
        group_denoise = QHBoxLayout()
        group_denoise.addWidget(QLabel("Denoising Strength:"))
        self.slider_denoise = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(40) # 0.4
        self.spin_denoise = QSpinBox()
        self.spin_denoise.setRange(0, 100)
        self.spin_denoise.setValue(40)
        
        # 슬라이더-스핀박스 연동
        self.slider_denoise.valueChanged.connect(self.spin_denoise.setValue)
        self.spin_denoise.valueChanged.connect(self.slider_denoise.setValue)
        
        group_denoise.addWidget(self.slider_denoise)
        group_denoise.addWidget(self.spin_denoise)
        layout.addLayout(group_denoise)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_detect_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 탐지 설정
        group_det = QGroupBox("Detection Settings")
        l_det = QVBoxLayout()
        
        l_det.addWidget(QLabel("Detector Model:"))
        self.combo_detector = QComboBox()
        self.combo_detector.addItems(["face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt"])
        l_det.addWidget(self.combo_detector)
        
        l_det.addWidget(QLabel("Confidence Threshold (0.0 ~ 1.0):"))
        self.spin_conf = QSpinBox() # 소수점 대신 0~100 정수로 처리 후 변환
        self.spin_conf.setRange(1, 100)
        self.spin_conf.setValue(35) # 0.35
        l_det.addWidget(self.spin_conf)

        self.chk_anatomy = QCheckBox("🧟 Anatomy Check (괴물 얼굴 필터링)")
        self.chk_anatomy.setChecked(True)
        self.chk_anatomy.setStyleSheet("color: #ffab91; font-weight: bold;")
        l_det.addWidget(self.chk_anatomy)
        
        group_det.setLayout(l_det)
        layout.addWidget(group_det)
        
        # 기하학 설정
        group_geo = QGroupBox("Geometry & Rotation")
        l_geo = QVBoxLayout()
        
        self.chk_rotate = QCheckBox("🔄 Auto Rotation Correction (누운 얼굴 보정)")
        self.chk_rotate.setChecked(True)
        self.chk_rotate.setStyleSheet("color: #80cbc4; font-weight: bold;")
        self.chk_rotate.setToolTip("활성화 시: 눈 좌표를 계산하여 0도(정자세)로 회전시킨 후 인페인팅합니다.")
        l_geo.addWidget(self.chk_rotate)
        
        l_geo.addWidget(QLabel("Crop Padding (여백 비율 %):"))
        self.spin_padding = QSpinBox()
        self.spin_padding.setRange(0, 100)
        self.spin_padding.setValue(25) # 0.25
        l_geo.addWidget(self.spin_padding)
        
        group_geo.setLayout(l_geo)
        layout.addWidget(group_geo)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_advanced_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # ControlNet
        group_cn = QGroupBox("ControlNet & Guidance")
        l_cn = QVBoxLayout()
        
        self.chk_controlnet = QCheckBox("Use ControlNet Tile (형태 유지)")
        self.chk_controlnet.setChecked(True)
        l_cn.addWidget(self.chk_controlnet)
        
        l_cn.addWidget(QLabel("Guidance Start (Step %):"))
        self.slider_g_start = QSlider(Qt.Orientation.Horizontal)
        self.slider_g_start.setRange(0, 100)
        self.slider_g_start.setValue(0)
        l_cn.addWidget(self.slider_g_start)

        l_cn.addWidget(QLabel("Guidance End (Step %):"))
        self.slider_g_end = QSlider(Qt.Orientation.Horizontal)
        self.slider_g_end.setRange(0, 100)
        self.slider_g_end.setValue(100) # 1.0 (끝까지)
        self.slider_g_end.setToolTip("값을 낮추면(예: 40) 후반부에는 AI가 자유롭게 그립니다.")
        l_cn.addWidget(self.slider_g_end)
        
        group_cn.setLayout(l_cn)
        layout.addWidget(group_cn)
        
        # Metadata
        self.chk_metadata = QCheckBox("💾 Save Metadata (Civitai/WebUI Compatible)")
        self.chk_metadata.setChecked(True)
        layout.addWidget(self.chk_metadata)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    # =========================================================
    # 로직 및 이벤트 핸들러
    # =========================================================
    def select_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if path:
            self.lbl_input.setText(path)
            self.lbl_input.setStyleSheet("color: #00e676; font-weight: bold;")

    def log(self, message):
        self.log_view.append(message)
        # 스크롤 자동 이동
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_processing(self):
        input_path = self.lbl_input.text()
        if not os.path.isdir(input_path):
            self.log("[Error] 유효한 입력 폴더를 선택해주세요.")
            return

        # 1. 파일 목록 스캔
        exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(exts)]
        
        if not files:
            self.log("[Error] 폴더에 처리할 이미지가 없습니다.")
            return

        self.log(f"[Info] 총 {len(files)}장의 이미지를 처리 대기열에 등록합니다.")
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(files))
        
        # 2. Config 객체 생성 (UI 값 반영)
        config = DetailerConfig(
            checkpoint_file=self.combo_ckpt.currentText(),
            pos_prompt=self.txt_pos.toPlainText(),
            neg_prompt=self.txt_neg.toPlainText(),
            denoising_strength=self.slider_denoise.value() / 100.0,
            
            # Detection Tab
            detector_model=self.combo_detector.currentText(),
            conf_thresh=self.spin_conf.value() / 100.0,
            anatomy_check=self.chk_anatomy.isChecked(),
            auto_rotate=self.chk_rotate.isChecked(),
            crop_padding=self.spin_padding.value() / 100.0,
            
            # Advanced Tab
            use_controlnet=self.chk_controlnet.isChecked(),
            guidance_start=self.slider_g_start.value() / 100.0,
            guidance_end=self.slider_g_end.value() / 100.0
        )
        
        # 시스템 설정 업데이트
        self.sys_config.save_metadata = self.chk_metadata.isChecked()

        # 3. 큐 채우기
        for f in files:
            self.task_queue.put((f, config)) # (경로, 설정) 튜플 저장

        # 4. 워커 스레드 시작 (Dual GPU Strategy)
        self.workers = []
        gpu_count = 1
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
        
        self.log(f"[Info] {gpu_count}개의 GPU 워커를 가동합니다.")
        
        for i in range(gpu_count):
            # 큐를 공유하는 워커 생성
            worker = ProcessWorker(device_id=i, task_queue=self.task_queue, sys_config=self.sys_config)
            worker.log_msg.connect(self.log)
            worker.result_ready.connect(self.update_progress)
            worker.start()
            self.workers.append(worker)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_processing(self):
        self.log("[Info] 작업 중단 요청됨. 현재 작업까지만 완료하고 멈춥니다.")
        # 큐 비우기
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                break
        
        # 워커 중지
        for w in self.workers:
            w.stop()
        
        self.workers.clear()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_progress(self, filename):
        val = self.progress_bar.value() + 1
        self.progress_bar.setValue(val)
        self.log(f"[Complete] {filename} 처리 완료 ({val}/{self.progress_bar.maximum()})")
        
        if val >= self.progress_bar.maximum():
            self.log("[System] 모든 작업이 완료되었습니다! 🎉")
            self.stop_processing()

    def apply_dark_theme(self):
        """다크 테마 스타일시트 적용"""
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QWidget { background-color: #2b2b2b; color: #ffffff; }
            QGroupBox { 
                border: 1px solid #555; 
                margin-top: 10px; 
                font-weight: bold;
                border-radius: 5px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { 
                background-color: #3c3c3c; 
                border: 1px solid #555; 
                border-radius: 4px; 
                padding: 5px;
                color: #fff;
            }
            QPushButton:hover { background-color: #505050; }
            QTextEdit, QLineEdit, QComboBox, QSpinBox {
                background-color: #1e1e1e; 
                border: 1px solid #3c3c3c; 
                color: #eee;
                border-radius: 3px;
                padding: 2px;
            }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab {
                background: #333;
                color: #aaa;
                padding: 8px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #444;
                color: #fff;
                font-weight: bold;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())