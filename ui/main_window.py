import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QLabel, QPushButton, QSplitter,
                             QTextEdit, QComboBox, QGroupBox, QFileDialog, QSizePolicy, QGridLayout,
                             QMenu, QMessageBox, QProgressBar, QSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup

from ui.main_window_tabs import AdetailerUnitWidget
from ui.workers import ProcessingController
from ui.components import ImageCanvas, ComparisonViewer, FileQueueWidget
from core.config import config_instance as cfg
from ui.styles import ModernTheme

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone ADetailer - Dual GPU Edition")
        # [Ref] Increase Window Size for Wide Right Panel View (User Permission)
        self.resize(1805, 1560) # Wide HD+
        
        # [Diagnostic] Multi-GPU check (Local Import for Isolation)
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"[System] Multi-GPU Detection: {count} GPUs found.")
            for i in range(count):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("[System] No GPU detected by PyTorch.")

        self.controller = None
        self.preview_processor = None # [New] For quick detection preview
        
        # [Fix] Theme Initialization (Load from Config)
        self.current_theme = "light" # Default before init_ui
        
        self.init_ui()
        self.load_theme_setting()

    def init_ui(self):
        # ============================================================
        # [Menu Bar] 파일 메뉴 & 테마 메뉴
        # ============================================================
        menubar = self.menuBar()
        menubar.clear()  # [Fix] Prevent duplicate menus if init_ui called twice
        
        # [File Menu]
        file_menu = menubar.addMenu('파일 (File)')
        
        action_save_all = QAction('전체 설정 저장 (Save All Configs)', self)
        action_save_all.triggered.connect(self.save_all_configs)
        file_menu.addAction(action_save_all)
        
        action_save_current = QAction('현재 탭 설정 저장 (Save Current Tab)', self)
        action_save_current.triggered.connect(self.save_current_tab_config)
        file_menu.addAction(action_save_current)
        
        file_menu.addSeparator()
        action_exit = QAction('종료 (Exit)', self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # [View Menu]
        view_menu = menubar.addMenu('보기 (View)')
        theme_menu = view_menu.addMenu('테마 (Theme)')
        
        theme_group = QActionGroup(self)
        self.action_dark = QAction('다크 모드 (Dark)', self, checkable=True)
        self.action_dark.triggered.connect(self.apply_dark_theme)
        theme_group.addAction(self.action_dark)
        theme_menu.addAction(self.action_dark)
        
        self.action_light = QAction('라이트 모드 (Light)', self, checkable=True)
        self.action_light.triggered.connect(self.apply_light_theme)
        theme_group.addAction(self.action_light)
        self.action_light.setChecked(True)
        theme_menu.addAction(self.action_light)
        
        # ============================================================
        # [Main Layout] Splitter 적용 (좌우 조절 가능)
        # ============================================================
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)

        # ============================================================
        # [Left Panel] Settings
        # ============================================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Global Model Settings
        self.global_group = QGroupBox("🛠️ 기본 모델 설정 (Global)")
        global_layout = QGridLayout()
        
        self.combo_global_ckpt = QComboBox()
        ckpt_dir = cfg.get_path('checkpoint')
        if ckpt_dir and os.path.exists(ckpt_dir):
            self.combo_global_ckpt.addItems([f for f in os.listdir(ckpt_dir) if f.endswith(('.ckpt', '.safetensors'))])
        else:
            self.combo_global_ckpt.addItem("No Checkpoints Found")

        self.combo_global_vae = QComboBox()
        vae_dir = cfg.get_path('vae')
        if vae_dir and os.path.exists(vae_dir):
            self.combo_global_vae.addItem("Automatic")
            self.combo_global_vae.addItems([f for f in os.listdir(vae_dir) if f.endswith(('.pt', '.ckpt', '.safetensors'))])
        else:
            self.combo_global_vae.addItem("Automatic")
        
        # [New] Global Save/Load Buttons (Shortened for Single Row)
        btn_global_save = QPushButton("💾")
        btn_global_save.setToolTip("현재 모든 설정(모델, 탭 설정 등)을 config.yaml에 저장합니다.")
        btn_global_save.clicked.connect(self.save_global_settings)
        btn_global_save.setFixedSize(30, 30) # Compact Icon Button
        
        btn_global_load = QPushButton("🔄")
        btn_global_load.setToolTip("config.yaml에서 설정을 다시 불러옵니다.")
        btn_global_load.clicked.connect(self.load_global_settings)
        btn_global_load.setFixedSize(30, 30) # Compact Icon Button

        # [Ref] Single-Row Layout for Global Settings
        # Layout: [Ckpt Label] [Ckpt Combo] [Save] | [VAE Label] [VAE Combo] [Load]
        # To fit in 1/3 width (~600px), we use minimal spacing
        
        global_layout = QHBoxLayout()
        global_layout.setContentsMargins(5, 5, 5, 5)
        global_layout.setSpacing(5)
        
        # Checkpoint Section
        lbl_ckpt = QLabel("Ckpt:")
        lbl_ckpt.setToolTip("Stable Diffusion Checkpoint Model")
        global_layout.addWidget(lbl_ckpt)
        
        self.combo_global_ckpt.setMaximumWidth(220) # [Fix] Limit width
        global_layout.addWidget(self.combo_global_ckpt, 1) # Stretch 1
        
        btn_global_save.setFixedSize(40, 30) # [Fix] Slightly wider
        global_layout.addWidget(btn_global_save)
        
        # VAE Section
        # Divider (Vertical Line)
        line = QLabel("|")
        line.setStyleSheet("color: gray;")
        global_layout.addWidget(line)
        
        lbl_vae = QLabel("VAE:")
        lbl_vae.setToolTip("VAE Model")
        global_layout.addWidget(lbl_vae)
        
        self.combo_global_vae.setMaximumWidth(220) # [Fix] Limit width
        global_layout.addWidget(self.combo_global_vae, 1) # Stretch 1
        
        btn_global_load.setFixedSize(40, 30) # [Fix] Slightly wider
        global_layout.addWidget(btn_global_load)
        
        self.global_group.setLayout(global_layout)
        left_layout.addWidget(self.global_group)

        # 2. Tabs
        self.tabs = QTabWidget()
        self.unit_widgets = []
        
        max_passes = cfg.get('system', 'max_passes') or 15
        
        for i in range(1, max_passes + 1): 
            # 페이지 생성
            tab = AdetailerUnitWidget(unit_name=f"패스 {i}")
            # [New] Connect Preview Signal
            tab.preview_requested.connect(self.on_detect_preview_requested)
            
            self.unit_widgets.append(tab)
            self.tabs.addTab(tab, f"패스 {i}")
            
        # [Fix] Force select first tab (Pass 1) on startup
        self.tabs.setCurrentIndex(0)
        
        left_layout.addWidget(self.tabs)
        
        left_panel.setMinimumWidth(400) # 최소 너비 확보 (40% 비율 유연성)

        # [Ref] Splitter Ratio (1:2 => Left Panel ~1/3 Width)
        # Main Window Width is set to 1800, so Left ~600, Right ~1200
        self.splitter.setSizes([600, 1200])

        # ============================================================
        # [Right Panel] Preview & Logs
        # ============================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(5)

        # 1. Preview
        self.sub_view = ImageCanvas()
        self.sub_view.setMinimumHeight(300)

        # 2. Comparison
        self.compare_view = ComparisonViewer()
        self.compare_view.setMinimumHeight(400)

        # 3. Queue
        self.file_queue = FileQueueWidget()
        self.file_queue.setMinimumHeight(200)
        self.file_queue.file_clicked.connect(self.on_file_clicked)
        # [New] Connect Rotation Signal
        self.file_queue.file_rotated.connect(lambda p, a: self.on_file_clicked(p))

        # 4. Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        # 5. Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("📁 이미지 불러오기")
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_load.setFixedHeight(45)
        
        self.btn_run = QPushButton("🚀 일괄 실행 (Run Batch)")
        self.btn_run.clicked.connect(self.start_processing)
        self.btn_run.setFixedHeight(45)
        self.btn_run.setFixedHeight(45)
        self.btn_run.setProperty("class", "action-button-run") # For future specific styling if needed
        # [Ref] Removed inline style, use theme defaults or add to styles.py if specific class needed
        # Keeping minimal inline for specific color semantic (Run = Green) but simplified
        self.btn_run.setStyleSheet(f"""
            QPushButton {{ 
                background-color: #27ae60; color: white; font-weight: bold; border: none; 
            }}
            QPushButton:hover {{ background-color: #2ecc71; }}
            QPushButton:pressed {{ background-color: #219150; }}
        """)
        
        self.btn_stop = QPushButton("⏹ 중지")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setFixedHeight(45)
        
        # [New] Worker Count Control
        l_worker = QVBoxLayout()
        l_worker.setSpacing(0)
        lbl_worker = QLabel("프로세스 수:")
        lbl_worker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_worker.setStyleSheet("font-size: 10px; color: #bdc3c7;")
        
        self.spin_worker_count = QSpinBox()
        self.spin_worker_count.setRange(1, 16)
        # Default: GPU Count or 1 (Local Import for Isolation)
        import torch
        default_workers = 1
        if torch.cuda.is_available():
            default_workers = torch.cuda.device_count()
        self.spin_worker_count.setValue(default_workers)
        self.spin_worker_count.setToolTip("병렬 처리를 위한 워커 프로세스 수입니다.\n1 GPU에서도 여러 워커를 사용하여 속도를 높일 수 있습니다 (VRAM 주의).")
        self.spin_worker_count.setFixedHeight(25)
        self.spin_worker_count.setStyleSheet("font-weight: bold;")
        
        l_worker.addWidget(lbl_worker)
        l_worker.addWidget(self.spin_worker_count)
        
        # [Adjust] Reordered buttons: WorkerControl -> Load -> Run -> Stop
        btn_layout.addLayout(l_worker) # Add Worker Control (First)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)

        # [New] Splitter for Preview and Compare
        # [New] Splitter for Preview and Compare
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.addWidget(self.sub_view)
        self.right_splitter.addWidget(self.compare_view)
        # [Adjust] Use setSizes instead of stretch factor for manual control
        # Example: [Preview Height, Compare Height] in pixels
        self.right_splitter.setSizes([400, 500]) 

        # Progress Bar (Moved to Right Panel Bottom)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(True) # Always visible now
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {ModernTheme.DARK_BORDER if self.current_theme == 'dark' else ModernTheme.LIGHT_BORDER};
                border-radius: 4px;
                text-align: center;
                background-color: transparent;
            }}
            QProgressBar::chunk {{
                background-color: {ModernTheme.DARK_ACCENT if self.current_theme == 'dark' else ModernTheme.LIGHT_ACCENT};
                border-radius: 3px;
            }}
        """)

        right_layout.addWidget(self.right_splitter, 3)
        right_layout.addWidget(self.file_queue, 1)
        right_layout.addWidget(self.log_text, 0)
        right_layout.addLayout(btn_layout)
        right_layout.addWidget(self.progress_bar) # [New Position]
        
        # Add to Splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        
        # [New] Apply Splitter Size (Force Left Limit)
        # Using QTimer to apply AFTER layout calculation
        # [Adjust] Widen Left Panel (750px) for better visibility
        # [New] Apply Splitter Size (Force Left Limit)
        # Using QTimer to apply AFTER layout calculation
        # [Adjust] Widen Left Panel (1070px) for better visibility
        QTimer.singleShot(0, lambda: self.splitter.setSizes([1070, 735]))

        self.status_filename_label = QLabel("")
        self.status_filename_label.setStyleSheet("margin-left: 10px;")
        self.statusBar().addPermanentWidget(self.status_filename_label)
        # self.statusBar().addPermanentWidget(self.progress_bar) # Removed from status bar
        self.statusBar().showMessage("[System] Initialized. Ready.")

        # Trigger initial model check
        if self.combo_global_ckpt.count() > 0:
            self.on_global_ckpt_changed(self.combo_global_ckpt.currentText())
            
        # [New] 초기 실행 시 config.yaml 값 로드
        self.load_global_settings(silent=True)

    # --- Save Logic ---
    def save_all_configs(self):
        """모든 탭의 설정을 config.yaml에 저장"""
        all_settings = {}
        for i, tab in enumerate(self.unit_widgets):
            all_settings[tab.unit_name] = tab.get_config()
        
        # 'ui_settings' 키 아래에 저장하여 시스템 설정과 분리
        success = cfg.save_config({'ui_settings': all_settings})
        if success:
            self.log("[Config] All tab settings saved to config.yaml")
            QMessageBox.information(self, "저장 완료", "모든 탭 설정이 config.yaml에 저장되었습니다.")
        else:
            self.log("[Config] Failed to save settings.")

    def save_current_tab_config(self):
        """현재 선택된 탭의 설정만 저장"""
        current_idx = self.tabs.currentIndex()
        if current_idx < 0: return
        
        tab = self.unit_widgets[current_idx]
        current_config = tab.get_config()
        
        # 기존 설정 로드 후 업데이트
        existing_ui_settings = cfg.get('ui_settings') or {}
        existing_ui_settings[tab.unit_name] = current_config
        
        success = cfg.save_config({'ui_settings': existing_ui_settings})
        if success:
            self.log(f"[Config] Settings for {tab.unit_name} saved.")
            QMessageBox.information(self, "저장 완료", f"{tab.unit_name} 설정이 저장되었습니다.")

    def save_global_settings(self):
        """글로벌 모델 설정 및 모든 탭의 설정을 config.yaml에 저장"""
        # 1. Global Settings
        ckpt = self.combo_global_ckpt.currentText()
        vae = self.combo_global_vae.currentText()
        workers = self.spin_worker_count.value()
        
        files_conf = cfg.get('files') or {}
        files_conf['checkpoint_file'] = ckpt
        files_conf['vae_file'] = vae
        
        system_conf = cfg.get('system') or {}
        system_conf = cfg.get('system') or {}
        system_conf['worker_count'] = workers
        system_conf['theme'] = self.current_theme # Save Theme
        
        # 2. Tab Settings (UI Settings)
        all_settings = {}
        for i, tab in enumerate(self.unit_widgets):
            all_settings[tab.unit_name] = tab.get_config()
        
        # 3. Save All
        data_to_save = {
            'files': files_conf,
            'system': system_conf,
            'ui_settings': all_settings
        }
        
        if cfg.save_config(data_to_save):
            self.log(f"[Config] All settings saved: CKPT='{ckpt}', VAE='{vae}', and {len(all_settings)} tabs.")
            QMessageBox.information(self, "저장 완료 (Saved)", "모든 설정(글로벌 모델 + 탭 설정)이 저장되었습니다.")
        else:
            self.log("[Config] Failed to save settings.")

    def load_global_settings(self, silent=False):
        """config.yaml에서 글로벌 모델 설정을 불러와 UI에 적용"""
        cfg.load_config(cfg.config_path)
        
        ckpt = cfg.get('files', 'checkpoint_file')
        vae = cfg.get('files', 'vae_file')
        workers = cfg.get('system', 'worker_count')
        
        if ckpt:
            idx = self.combo_global_ckpt.findText(ckpt)
            if idx >= 0: self.combo_global_ckpt.setCurrentIndex(idx)
        if vae:
            idx = self.combo_global_vae.findText(vae)
            if idx >= 0: self.combo_global_vae.setCurrentIndex(idx)
            
        if workers and isinstance(workers, int):
            self.spin_worker_count.setValue(workers)
        if workers and isinstance(workers, int):
            self.spin_worker_count.setValue(workers)
            
        if not silent:
            self.log(f"[Config] Global settings loaded: CKPT='{ckpt}', VAE='{vae}', Workers={workers}")
            QMessageBox.information(self, "로드 완료", "글로벌 모델 설정을 불러왔습니다.")
            
    def load_theme_setting(self):
        """저장된 테마 설정 로드 및 적용"""
        saved_theme = cfg.get('system', 'theme') or "light"
        if saved_theme == "dark":
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    # --- Theme & Basics ---
    def apply_dark_theme(self):
        self.current_theme = "dark"
        self.setStyleSheet(ModernTheme.get_dark_theme())
        
        # [Ref] Update specifics that depend on theme variables but aren't covered by global sheet
        self.log_text.setStyleSheet(f"background-color: {ModernTheme.DARK_BG_INPUT}; color: #00ff00; border: 1px solid {ModernTheme.DARK_BORDER}; font-family: Consolas;")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; border: none; font-weight: bold;")
        
        # Update progress bar dynamically
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid {ModernTheme.DARK_BORDER}; text-align: center; color: {ModernTheme.DARK_TEXT_MAIN}; }}
            QProgressBar::chunk {{ background-color: {ModernTheme.DARK_ACCENT}; }}
        """)

        self.sub_view.set_theme("dark")
        self.compare_view.set_theme("dark")
        self.file_queue.set_theme("dark")
        
        # [New] Apply theme to all tabs (for Graph)
        for tab in self.unit_widgets:
            tab.set_theme("dark")

    def apply_light_theme(self):
        self.current_theme = "light"
        self.setStyleSheet(ModernTheme.get_light_theme())
        
        # [Ref] Specifics for Light
        self.log_text.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BG_INPUT}; color: #333; border: 1px solid {ModernTheme.LIGHT_BORDER}; font-family: Consolas;")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; border: none; font-weight: bold;")
        
        # Update progress bar dynamically
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid {ModernTheme.LIGHT_BORDER}; text-align: center; color: {ModernTheme.LIGHT_TEXT_MAIN}; }}
            QProgressBar::chunk {{ background-color: {ModernTheme.LIGHT_ACCENT}; }}
        """)

        self.sub_view.set_theme("light")
        self.compare_view.set_theme("light")
        self.file_queue.set_theme("light")

        # [New] Apply theme to all tabs (for Graph)
        for tab in self.unit_widgets:
            tab.set_theme("light")

    def log(self, message):
        self.log_text.append(message)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def load_image_dialog(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if fnames:
            for f in fnames:
                self.file_queue._add_item(f)
            self.log(f"Added {len(fnames)} files to queue.")

    def on_file_clicked(self, file_path):
        try:
            stream = open(file_path.encode("utf-8"), "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img_before = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            
            # [New] Apply Rotation
            angle = self.file_queue.get_rotation(file_path)
            if angle != 0 and img_before is not None:
                if angle == 90 or angle == -270:
                    img_before = cv2.rotate(img_before, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180 or angle == -180:
                    img_before = cv2.rotate(img_before, cv2.ROTATE_180)
                elif angle == 270 or angle == -90:
                    img_before = cv2.rotate(img_before, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if img_before is not None:
                # [Fix] 결과물이 존재하면 로드하여 After 이미지로 설정 (슬라이더 작동 보장)
                output_dir = cfg.get('system', 'output_path') or "outputs"
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, filename)
                
                img_after = img_before # 기본값은 원본
                if os.path.exists(output_path):
                    try:
                        stream_out = open(output_path.encode("utf-8"), "rb")
                        bytes_out = bytearray(stream_out.read())
                        numpyarray_out = np.asarray(bytes_out, dtype=np.uint8)
                        loaded_after = cv2.imdecode(numpyarray_out, cv2.IMREAD_COLOR)
                        if loaded_after is not None:
                            img_after = loaded_after
                    except:
                        pass

                self.compare_view.set_images(img_before, img_after)
                self.sub_view.set_image(img_after)
        except Exception as e:
            self.log(f"Error loading preview: {e}")

    def set_ui_enabled(self, enabled):
        """처리 중 UI 활성화/비활성화 제어"""
        self.btn_load.setEnabled(enabled)
        self.btn_run.setEnabled(enabled)
        self.tabs.setEnabled(enabled)
        self.global_group.setEnabled(enabled)
        self.file_queue.setEnabled(enabled)
        
        # 중지 버튼은 반대로 동작 (실행 중일 때만 활성화)
        self.btn_stop.setEnabled(not enabled)
        self.btn_stop.setStyleSheet("background-color: #d32f2f; color: white;" if not enabled else "background-color: #cccccc; color: #666666;")

    def start_processing(self):
        # [Fix] Get tasks with rotation info
        tasks = self.file_queue.get_all_tasks() # [(path, angle), ...]
        if not tasks:
            self.log("No files to process.")
            return

        # [Fix] 글로벌 모델 설정 수집
        global_ckpt = self.combo_global_ckpt.currentText()
        global_vae = self.combo_global_vae.currentText()

        configs = []
        for tab in self.unit_widgets:
            cfg_data = tab.get_config()
            if cfg_data['enabled']:
                # 각 탭 설정에 글로벌 설정 주입
                cfg_data['global_ckpt_name'] = global_ckpt
                cfg_data['global_vae_name'] = global_vae
                # [Fix] 로그 가시성을 위해 패스 이름 주입
                cfg_data['unit_name'] = tab.unit_name
                configs.append(cfg_data)

        if not configs:
            self.log("No enabled tabs. Enable at least one pass.")
            return

        # [Fix] 중복 실행 방지: 기존 작업 중지
        if self.controller:
            self.controller.stop()

        # UI 비활성화 (중지 버튼 제외)
        self.set_ui_enabled(False)

        self.log("Starting batch processing...")
        self.controller = ProcessingController(tasks, configs)
        self.controller.log_signal.connect(self.log)
        self.controller.progress_signal.connect(self.update_progress)
        self.controller.file_started_signal.connect(self.update_status_filename)
        self.controller.preview_signal.connect(self.update_preview)
        self.controller.result_signal.connect(self.handle_result)
        
        # [Fix] Pass Worker Count
        workers = self.spin_worker_count.value()
        self.controller.start_processing(max_workers=workers)

    def handle_result(self, path, result_img):
        if result_img is None:
            self.log(f"Failed: {os.path.basename(path)}")
            return

        self.log(f"Finished: {os.path.basename(path)}")
        self.file_queue.select_item_by_path(path)
        
        # [Fix] 처리 완료 시 원본 이미지도 함께 로드하여 비교 뷰어(슬라이더) 즉시 갱신
        try:
            stream = open(path.encode("utf-8"), "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img_before = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            
            # [New] Apply Rotation for Display
            angle = self.file_queue.get_rotation(path)
            if angle != 0:
                # Rotate function
                def rotate_img(img, a):
                    if img is None: return None
                    if a == 90 or a == -270: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif a == 180 or a == -180: return cv2.rotate(img, cv2.ROTATE_180)
                    elif a == 270 or a == -90: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    return img
                
                img_before = rotate_img(img_before, angle)
                result_img = rotate_img(result_img, angle) # result_img is likely restored to original, so rotate it back for view
            
            self.compare_view.set_images(img_before, result_img)
        except:
            # Fallback if load fails
            self.compare_view.pixmap_after = self.compare_view._np2pix(result_img)
            self.compare_view.update()
            
        self.sub_view.set_image(result_img)

    def update_progress(self, current, total):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        if current >= total:
            self.progress_bar.setVisible(False)
            self.status_filename_label.setText("")
            # 모든 작업 완료 시 UI 다시 활성화
            self.set_ui_enabled(True)

    def update_status_filename(self, filename):
        self.status_filename_label.setText(f"Processing: {filename}")

    def update_preview(self, img):
        self.sub_view.set_image(img)

    def stop_processing(self):
        self.log("Stopping processing...")
        if self.controller:
            self.controller.stop()
        # 중지 시 UI 다시 활성화
        self.set_ui_enabled(True)

    def on_global_ckpt_changed(self, text):
        """글로벌 모델 변경 시 각 탭에 알림 (UI 동적 업데이트)"""
        for tab in self.unit_widgets:
            tab.on_global_model_changed(text)

    def on_detect_preview_requested(self, config):
        """[New] 탭에서 탐지 미리보기 요청 시 처리"""
        # 1. Get Selected Image
        # Assuming FileQueueWidget has a way to get selected or we use the last loaded/viewed.
        # MainWindow doesn't track 'selected' explicitly, but compare_view usually shows it.
        # But we need the filepath to load raw image.
        
        # Try to get from queue (need to check if FileQueueWidget exposes selection)
        # Let's rely on self.last_clicked_path if stored, or ask queue.
        # Actually FileQueueWidget is simple. Let's assume user clicked an image and it's visible.
        # But for accurate testing, we should get the CURRENTLY SELECTED item in the list.
        
        # Access ListWidget directly?
        items = self.file_queue.list_widget.selectedItems()
        if not items:
            QMessageBox.warning(self, "이미지 없음", "탐지할 이미지를 목록에서 선택해 주세요.")
            return
            
        file_path = items[0].data(Qt.ItemDataRole.UserRole) # Assuming path stored in UserRole or just use tool tip / text
        # FileQueueWidget._add_item stores path in UserRole? Let's check or assume text is filename?
        # Actually FileQueueWidget usually stores path.
        # Let's try to infer from text if UserRole fails, but usually we just store path.
        # Wait, I don't see FileQueueWidget impl here. 
        # But commonly we store full path.
        if not file_path: 
            # Fallback: file_path stored in item text? or UserRole.
            # Let's assume UserRole is used. If None, try config.system.output... No.
            # Re-read ui/main_window.py to see how on_file_clicked gets path.
            pass

        # 2. Lazy Init Processor
        if self.preview_processor is None:
            from core.pipeline import ImageProcessor
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.preview_processor = ImageProcessor(device=device, log_callback=self.log)
            
        # 3. Load Image
        try:
            stream = open(file_path.encode("utf-8"), "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            
            if img is None: raise Exception("Decode failed")
            
            # [New] Apply Rotation for Preview
            angle = self.file_queue.get_rotation(file_path)
            if angle != 0:
                if angle == 90 or angle == -270:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180 or angle == -180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif angle == 270 or angle == -90:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"이미지 로드 실패: {e}")
            return

        # 4. Run Detection
        self.set_ui_enabled(False) # Prevent other actions
        try:
            preview_img = self.preview_processor.detect_preview(img, config)
            self.sub_view.set_image(preview_img)
            self.log(f"[Preview] Detection finished for {os.path.basename(file_path)}")
        except Exception as e:
            self.log(f"[Error] Preview failed: {e}")
            QMessageBox.critical(self, "Error", f"탐지 오류: {e}")
        finally:
            self.set_ui_enabled(True)
            # [Fix] Clean up VRAM in GUI process after preview
            if self.preview_processor:
                try:
                    self.preview_processor.detector.offload_models()
                    if self.preview_processor.sam:
                        self.preview_processor.sam.unload_model()
                    import torch
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())