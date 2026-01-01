import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QLabel, QPushButton, QSplitter, 
                             QTextEdit, QComboBox, QGroupBox, QFileDialog, QSizePolicy,
                             QMenu, QMessageBox, QProgressBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup

from ui.main_window_tabs import AdetailerUnitWidget
from ui.workers import ProcessingController
from ui.components import ImageCanvas, ComparisonViewer, FileQueueWidget
from core.config import config_instance as cfg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone ADetailer - Dual GPU Edition")
        self.resize(1600, 1200) # 기본 사이즈
        
        self.controller = None
        
        self.init_ui()
        self.apply_light_theme() # 기본 테마

    def init_ui(self):
        # ============================================================
        # [Menu Bar] 파일 메뉴 & 테마 메뉴
        # ============================================================
        menubar = self.menuBar()
        
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
        global_layout = QHBoxLayout()
        
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
        
        global_layout.addWidget(QLabel("체크포인트:"))
        global_layout.addWidget(self.combo_global_ckpt, 2)
        global_layout.addWidget(QLabel("VAE:"))
        global_layout.addWidget(self.combo_global_vae, 1)
        
        self.combo_global_ckpt.currentTextChanged.connect(self.on_global_ckpt_changed)
        
        self.global_group.setLayout(global_layout)
        left_layout.addWidget(self.global_group)

        # 2. Tabs
        self.tabs = QTabWidget()
        self.unit_widgets = []
        max_passes = cfg.get('system', 'max_passes') or 15
        for i in range(1, max_passes + 1): 
            tab = AdetailerUnitWidget(unit_name=f"패스 {i}")
            self.unit_widgets.append(tab)
            self.tabs.addTab(tab, f"패스 {i}")
        
        left_layout.addWidget(self.tabs)
        left_panel.setMinimumWidth(550) # 최소 너비 확보

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

        # 4. Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        # 5. Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("📁 이미지 불러오기")
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_load.setMinimumHeight(40)
        
        self.btn_run = QPushButton("🚀 일괄 실행 (Run Batch)")
        self.btn_run.clicked.connect(self.start_processing)
        self.btn_run.setMinimumHeight(40)
        
        self.btn_stop = QPushButton("⏹ 중지")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setMinimumHeight(40)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)

        right_layout.addWidget(self.sub_view, 1)
        right_layout.addWidget(self.compare_view, 2)
        right_layout.addWidget(self.file_queue, 1)
        right_layout.addWidget(self.log_text, 0)
        right_layout.addLayout(btn_layout)

        # Add to Splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        # Progress Bar in Status Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("[System] Initialized. Ready.")

        # Trigger initial model check
        if self.combo_global_ckpt.count() > 0:
            self.on_global_ckpt_changed(self.combo_global_ckpt.currentText())

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

    # --- Theme & Basics ---
    def apply_dark_theme(self):
        dark_style = """
            QMainWindow, QWidget { background-color: #2b2b2b; color: #eeeeee; font-size: 10pt; }
            QSplitter::handle { background-color: #444; width: 4px; }
            QGroupBox { border: 1px solid #555; margin-top: 15px; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4dabf7; font-weight: bold; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { 
                background-color: #333; border: 1px solid #555; padding: 4px; border-radius: 3px; color: #eee;
            }
            QPushButton { background-color: #0078d7; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QRadioButton { spacing: 5px; color: #eeeeee; }
            QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 2px solid #666; background-color: #333; }
            QRadioButton::indicator:checked { background-color: #4dabf7; border-color: #4dabf7; }
            QRadioButton::indicator:unchecked:hover { border-color: #888; }
        """
        self.setStyleSheet(dark_style)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border: 2px solid #c0392b; font-family: Consolas;")
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;")

    def apply_light_theme(self):
        light_style = """
            QMainWindow, QWidget { background-color: #f5f5f5; color: #333333; font-size: 10pt; }
            QSplitter::handle { background-color: #ccc; width: 4px; }
            QGroupBox { border: 1px solid #cccccc; margin-top: 15px; border-radius: 4px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #0056b3; font-weight: bold; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { 
                background-color: #ffffff; border: 1px solid #cccccc; padding: 4px; border-radius: 3px; color: #333;
            }
            QPushButton { background-color: #0078d7; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QRadioButton { spacing: 5px; color: #333333; }
            QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 2px solid #999; background-color: #fff; }
            QRadioButton::indicator:checked { background-color: #0078d7; border-color: #0078d7; }
            QRadioButton::indicator:unchecked:hover { border-color: #555; }
        """
        self.setStyleSheet(light_style)
        self.log_text.setStyleSheet("background-color: #ffffff; color: #000000; border: 2px solid #c0392b; font-family: Consolas;")
        self.btn_stop.setStyleSheet("background-color: #d32f2f; color: white;")

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
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            if img is not None:
                self.compare_view.set_images(img, img)
                self.sub_view.set_image(img)
        except Exception as e:
            self.log(f"Error loading preview: {e}")

    def start_processing(self):
        files = self.file_queue.get_all_files()
        if not files:
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
                configs.append(cfg_data)

        if not configs:
            self.log("No enabled tabs. Enable at least one pass.")
            return

        # [Fix] 중복 실행 방지: 기존 작업 중지
        if self.controller:
            self.controller.stop()

        self.log("Starting batch processing...")
        self.controller = ProcessingController(files, configs)
        self.controller.log_signal.connect(self.log)
        self.controller.progress_signal.connect(self.update_progress)
        self.controller.preview_signal.connect(self.update_preview)
        self.controller.result_signal.connect(self.handle_result)
        self.controller.start_processing()

    def handle_result(self, path, result_img):
        if result_img is None:
            self.log(f"Failed: {os.path.basename(path)}")
            return

        self.log(f"Finished: {os.path.basename(path)}")
        self.file_queue.select_item_by_path(path)
        self.sub_view.set_image(result_img)
        self.compare_view.pixmap_after = self.compare_view._np2pix(result_img)
        self.compare_view.update()

    def update_progress(self, current, total):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        if current >= total:
            self.progress_bar.setVisible(False)

    def update_preview(self, img):
        self.sub_view.set_image(img)

    def stop_processing(self):
        self.log("Stopping processing...")
        if self.controller:
            self.controller.stop()

    def on_global_ckpt_changed(self, text):
        """글로벌 모델 변경 시 각 탭에 알림 (UI 동적 업데이트)"""
        for tab in self.unit_widgets:
            tab.on_global_model_changed(text)

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())