import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QStackedWidget, QButtonGroup, QLabel, QPushButton, QSplitter, 
                             QTextEdit, QComboBox, QGroupBox, QFileDialog, QSizePolicy, QGridLayout,
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
        self.resize(2390, 1885) # 기본 사이즈
        
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
        
        global_layout.addWidget(QLabel("체크포인트:"), 0, 0)
        global_layout.addWidget(self.combo_global_ckpt, 0, 1)
        global_layout.addWidget(QLabel("VAE:"), 0, 2)
        global_layout.addWidget(self.combo_global_vae, 0, 3)
        
        self.combo_global_ckpt.currentTextChanged.connect(self.on_global_ckpt_changed)

        # [New] Global Save/Load Buttons
        btn_global_save = QPushButton("💾 저장")
        btn_global_save.setToolTip("현재 선택된 체크포인트와 VAE를 config.yaml에 저장합니다.")
        btn_global_save.clicked.connect(self.save_global_settings)
        btn_global_save.setMaximumWidth(70)
        
        btn_global_load = QPushButton("🔄 로드")
        btn_global_load.setToolTip("config.yaml에서 설정을 다시 불러옵니다.")
        btn_global_load.clicked.connect(self.load_global_settings)
        btn_global_load.setMaximumWidth(70)

        global_layout.addWidget(btn_global_save, 0, 4)
        global_layout.addWidget(btn_global_load, 0, 5)
        
        # [Fix] 콤보박스 비율 50:50 강제 (컬럼 1과 3의 확장 비율을 1:1로 설정)
        global_layout.setColumnStretch(1, 1)
        global_layout.setColumnStretch(3, 1)
        
        self.global_group.setLayout(global_layout)
        left_layout.addWidget(self.global_group)

        # 2. Custom Tab Navigation (2-Story Layout)
        # [New] 탭 대신 버튼 그리드를 사용하여 2층 구조 구현
        nav_container = QWidget()
        nav_layout = QGridLayout(nav_container)
        # [Fix] 너비가 불필요하게 확장되지 않도록 설정
        nav_container.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(2)
        
        self.stack = QStackedWidget()
        self.unit_widgets = []
        self.nav_buttons = QButtonGroup(self)
        self.nav_buttons.setExclusive(True)
        
        max_passes = cfg.get('system', 'max_passes') or 15
        
        for i in range(1, max_passes + 1): 
            # 버튼 생성
            btn = QPushButton(f"패스 {i}")
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            self.nav_buttons.addButton(btn, i - 1)
            
            # 2층 구조 배치 (1~8: 1층, 9~15: 2층)
            row = 0 if i <= 8 else 1
            col = (i - 1) % 8
            nav_layout.addWidget(btn, row, col)
            
            # 페이지 생성
            tab = AdetailerUnitWidget(unit_name=f"패스 {i}")
            self.unit_widgets.append(tab)
            self.stack.addWidget(tab)
        
        # 버튼 클릭 시 페이지 전환 연결
        self.nav_buttons.idClicked.connect(self.stack.setCurrentIndex)
        
        # 첫 번째 탭 선택
        if self.nav_buttons.button(0):
            self.nav_buttons.button(0).setChecked(True)
        
        left_layout.addWidget(nav_container)
        left_layout.addWidget(self.stack)
        
        left_panel.setMinimumWidth(400) # 최소 너비 확보 (40% 비율 유연성)

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
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 6)
        
        # [Fix] 초기 실행 시 40:60 비율 강제 적용 (2390px 기준 956:1434)
        self.splitter.setSizes([956, 1434])

        # Progress Bar in Status Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)

        self.status_filename_label = QLabel("")
        self.status_filename_label.setStyleSheet("margin-left: 10px;")

        self.statusBar().addPermanentWidget(self.status_filename_label)
        self.statusBar().addPermanentWidget(self.progress_bar)
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
        current_idx = self.stack.currentIndex()
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
        """글로벌 모델 설정을 config.yaml에 저장"""
        ckpt = self.combo_global_ckpt.currentText()
        vae = self.combo_global_vae.currentText()
        
        files_conf = cfg.get('files') or {}
        files_conf['checkpoint_file'] = ckpt
        files_conf['vae_file'] = vae
        
        if cfg.save_config({'files': files_conf}):
            self.log(f"[Config] Global settings saved: CKPT='{ckpt}', VAE='{vae}'")
            QMessageBox.information(self, "저장 완료", "글로벌 모델 설정이 config.yaml에 저장되었습니다.")
        else:
            self.log("[Config] Failed to save global settings.")

    def load_global_settings(self, silent=False):
        """config.yaml에서 글로벌 모델 설정을 불러와 UI에 적용"""
        cfg.load_config(cfg.config_path)
        
        ckpt = cfg.get('files', 'checkpoint_file')
        vae = cfg.get('files', 'vae_file')
        
        if ckpt:
            idx = self.combo_global_ckpt.findText(ckpt)
            if idx >= 0: self.combo_global_ckpt.setCurrentIndex(idx)
        if vae:
            idx = self.combo_global_vae.findText(vae)
            if idx >= 0: self.combo_global_vae.setCurrentIndex(idx)
            
        if not silent:
            self.log(f"[Config] Global settings loaded: CKPT='{ckpt}', VAE='{vae}'")
            QMessageBox.information(self, "로드 완료", "글로벌 모델 설정을 불러왔습니다.")

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
            QPushButton { background-color: #444; color: white; border: 1px solid #555; padding: 6px; border-radius: 4px; }
            QPushButton:checked { background-color: #0078d7; font-weight: bold; border: 1px solid #0056b3; }
            QPushButton:hover:!checked { background-color: #555; }
            QRadioButton { spacing: 5px; color: #eeeeee; }
            QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 2px solid #666; background-color: #333; }
            QRadioButton::indicator:checked { background-color: #4dabf7; border-color: #4dabf7; }
            QRadioButton::indicator:unchecked:hover { border-color: #888; }
        """
        self.setStyleSheet(dark_style)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border: 2px solid #c0392b; font-family: Consolas;")
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;")
        self.sub_view.set_theme("dark")
        self.compare_view.set_theme("dark")
        self.file_queue.set_theme("dark")

    def apply_light_theme(self):
        light_style = """
            QMainWindow, QWidget { background-color: #f5f5f5; color: #333333; font-size: 10pt; }
            QSplitter::handle { background-color: #ccc; width: 4px; }
            QGroupBox { border: 1px solid #cccccc; margin-top: 15px; border-radius: 4px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #0056b3; font-weight: bold; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { 
                background-color: #ffffff; border: 1px solid #cccccc; padding: 4px; border-radius: 3px; color: #333;
            }
            QPushButton { background-color: #f0f0f0; color: #333; border: 1px solid #ccc; padding: 6px; border-radius: 4px; }
            QPushButton:checked { background-color: #0078d7; color: white; font-weight: bold; border: 1px solid #0056b3; }
            QPushButton:hover:!checked { background-color: #e0e0e0; }
            QRadioButton { spacing: 5px; color: #333333; }
            QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 2px solid #999; background-color: #fff; }
            QRadioButton::indicator:checked { background-color: #0078d7; border-color: #0078d7; }
            QRadioButton::indicator:unchecked:hover { border-color: #555; }
        """
        self.setStyleSheet(light_style)
        self.log_text.setStyleSheet("background-color: #ffffff; color: #000000; border: 2px solid #c0392b; font-family: Consolas;")
        self.btn_stop.setStyleSheet("background-color: #d32f2f; color: white;")
        self.sub_view.set_theme("light")
        self.compare_view.set_theme("light")
        self.file_queue.set_theme("light")

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
        self.stack.setEnabled(enabled)
        self.global_group.setEnabled(enabled)
        self.file_queue.setEnabled(enabled)
        
        # 중지 버튼은 반대로 동작 (실행 중일 때만 활성화)
        self.btn_stop.setEnabled(not enabled)
        self.btn_stop.setStyleSheet("background-color: #d32f2f; color: white;" if not enabled else "background-color: #cccccc; color: #666666;")

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
        self.controller = ProcessingController(files, configs)
        self.controller.log_signal.connect(self.log)
        self.controller.progress_signal.connect(self.update_progress)
        self.controller.file_started_signal.connect(self.update_status_filename)
        self.controller.preview_signal.connect(self.update_preview)
        self.controller.result_signal.connect(self.handle_result)
        self.controller.start_processing()

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
            self.compare_view.set_images(img_before, result_img)
        except:
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

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())