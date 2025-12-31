import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QLabel, QPushButton, QSplitter, 
                             QTextEdit, QComboBox, QGroupBox, QScrollArea,
                             QFileDialog, QProgressBar, QFrame, QSizePolicy,
                             QMenu)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QAction, QActionGroup

# 탭 위젯 & 컨트롤러 (기존 유지)
from ui.main_window_tabs import AdetailerUnitWidget
from ui.workers import ProcessingController

class ImageDropLabel(QLabel):
    """이미지 드래그 앤 드롭을 지원하는 라벨"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("이미지를 드래그하거나 불러오세요")
        self.setAcceptDrops(True)
        self.main_window = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if image_files and self.main_window:
            self.main_window.handle_dropped_files(image_files)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone ADetailer - Dual GPU Edition")
        self.resize(1600, 1600)
        
        self.file_paths = []
        self.controller = None
        
        self.init_ui()
        # 기본 테마: 다크 모드
        self.apply_dark_theme()

    def init_ui(self):
        # ============================================================
        # [Menu Bar] 테마 선택 메뉴 추가
        # ============================================================
        menubar = self.menuBar()
        file_menu = menubar.addMenu('파일 (File)')
        
        # 보기 메뉴 -> 테마 서브메뉴
        view_menu = menubar.addMenu('보기 (View)')
        theme_menu = view_menu.addMenu('테마 (Theme)')
        
        # 테마 액션 그룹 (하나만 선택 가능)
        theme_group = QActionGroup(self)
        
        self.action_dark = QAction('다크 모드 (Dark)', self, checkable=True)
        self.action_dark.setChecked(True)
        self.action_dark.triggered.connect(self.apply_dark_theme)
        theme_group.addAction(self.action_dark)
        theme_menu.addAction(self.action_dark)
        
        self.action_light = QAction('라이트 모드 (Light)', self, checkable=True)
        self.action_light.triggered.connect(self.apply_light_theme)
        theme_group.addAction(self.action_light)
        theme_menu.addAction(self.action_light)
        
        # --- Main Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ============================================================
        # [Left Panel] Settings
        # ============================================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Global Model Settings
        self.global_group = QGroupBox("🛠️ 기본 모델 설정 (Global Model Settings)")
        global_layout = QHBoxLayout()
        
        self.combo_global_ckpt = QComboBox()
        self.combo_global_ckpt.addItems(["henmix_real_v6b.safetensors", "sd_v1.5_pruned.ckpt"])
        self.combo_global_vae = QComboBox()
        self.combo_global_vae.addItems(["Automatic", "vae-ft-mse-840000.pt"])
        
        global_layout.addWidget(QLabel("체크포인트:"))
        global_layout.addWidget(self.combo_global_ckpt, 2)
        global_layout.addWidget(QLabel("VAE:"))
        global_layout.addWidget(self.combo_global_vae, 1)
        
        self.global_group.setLayout(global_layout)
        left_layout.addWidget(self.global_group)

        # 2. Tabs
        self.tabs = QTabWidget()
        self.unit_widgets = []
        for i in range(1, 10): 
            tab = AdetailerUnitWidget(unit_name=f"패스 {i}")
            self.unit_widgets.append(tab)
            self.tabs.addTab(tab, f"패스 {i}")
        
        left_layout.addWidget(self.tabs)
        left_panel.setMinimumWidth(650)
        left_panel.setMaximumWidth(750)

        # ============================================================
        # [Right Panel] Preview & Logs
        # ============================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # 1. Mask Preview
        self.mask_preview_label = QLabel("Mask Preview / Sub View")
        self.mask_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_preview_label.setMinimumHeight(400)

        # 2. Main Preview (Drop Area)
        self.main_preview_label = ImageDropLabel()
        self.main_preview_label.main_window = self
        self.main_preview_label.setMinimumHeight(600)
        self.main_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 3. Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)

        # 4. Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("📁 이미지 불러오기")
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_run = QPushButton("🚀 일괄 실행 (Run Batch)")
        self.btn_run.clicked.connect(self.start_processing)
        self.btn_stop = QPushButton("⏹ 중지")
        self.btn_stop.clicked.connect(self.stop_processing) # 기능 연결 필요
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)

        right_layout.addWidget(self.mask_preview_label, 1)
        right_layout.addWidget(self.main_preview_label, 2)
        right_layout.addWidget(self.log_text, 1)
        right_layout.addLayout(btn_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        self.statusBar().showMessage("[System] Initialized. Ready.")

    def apply_dark_theme(self):
        """다크 모드 스타일시트 적용 (가시성 최적화)"""
        dark_style = """
            QMainWindow, QWidget { background-color: #2b2b2b; color: #eeeeee; font-size: 13px; }
            QGroupBox { border: 1px solid #555; margin-top: 15px; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4dabf7; font-weight: bold; }
            
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { 
                background-color: #333; border: 1px solid #555; padding: 4px; border-radius: 3px; color: #eee;
            }
            QComboBox::drop-down { border: none; }
            
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; padding: 8px 12px; margin-right: 2px; color: #aaa; }
            QTabBar::tab:selected { background: #444; font-weight: bold; border-bottom: 2px solid #4dabf7; color: #fff; }
            
            QPushButton { background-color: #0078d7; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #1084e0; }
            QPushButton:pressed { background-color: #006cc1; }
            
            QScrollBar:vertical { background: #2b2b2b; width: 12px; }
            QScrollBar::handle:vertical { background: #555; min-height: 20px; border-radius: 6px; }
            
            QMenuBar { background-color: #2b2b2b; color: #eee; }
            QMenuBar::item:selected { background-color: #444; }
            QMenu { background-color: #333; border: 1px solid #555; }
            QMenu::item:selected { background-color: #0078d7; }
        """
        self.setStyleSheet(dark_style)
        
        # 프리뷰 영역 별도 스타일 (어두운 배경 유지)
        self.mask_preview_label.setStyleSheet("background-color: #1e1e1e; border: 2px solid #d4b106; color: #aaa;")
        self.main_preview_label.setStyleSheet("background-color: #1e1e1e; border: 2px dashed #2ea043; color: #aaa;")
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border: 2px solid #c0392b; font-family: Consolas;")
        
        # 버튼 스타일 재설정 (중지 버튼 빨간색 유지)
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;")

    def apply_light_theme(self):
        """라이트 모드 스타일시트 적용 (가시성 최적화)"""
        light_style = """
            QMainWindow, QWidget { background-color: #f5f5f5; color: #333333; font-size: 13px; }
            QGroupBox { border: 1px solid #cccccc; margin-top: 15px; border-radius: 4px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #0056b3; font-weight: bold; }
            
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { 
                background-color: #ffffff; border: 1px solid #cccccc; padding: 4px; border-radius: 3px; color: #333;
            }
            
            QTabWidget::pane { border: 1px solid #cccccc; background-color: #ffffff; }
            QTabBar::tab { background: #e0e0e0; padding: 8px 12px; margin-right: 2px; color: #555; }
            QTabBar::tab:selected { background: #ffffff; font-weight: bold; border-bottom: 2px solid #0078d7; color: #000; }
            
            QPushButton { background-color: #0078d7; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:pressed { background-color: #004080; }
            
            QScrollBar:vertical { background: #f0f0f0; width: 12px; }
            QScrollBar::handle:vertical { background: #cccccc; min-height: 20px; border-radius: 6px; }

            QMenuBar { background-color: #e0e0e0; color: #000; }
            QMenuBar::item:selected { background-color: #cccccc; }
            QMenu { background-color: #ffffff; border: 1px solid #cccccc; }
            QMenu::item:selected { background-color: #0078d7; color: white; }
        """
        self.setStyleSheet(light_style)
        
        # 프리뷰 영역 별도 스타일 (밝은 배경)
        self.mask_preview_label.setStyleSheet("background-color: #ffffff; border: 2px solid #d4b106; color: #555;")
        self.main_preview_label.setStyleSheet("background-color: #ffffff; border: 2px dashed #2ea043; color: #555;")
        self.log_text.setStyleSheet("background-color: #ffffff; color: #000000; border: 2px solid #c0392b; font-family: Consolas;")
        
        # 버튼 스타일 재설정
        self.btn_stop.setStyleSheet("background-color: #d32f2f; color: white;")

    def log(self, message):
        self.log_text.append(message)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def load_image_dialog(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if fnames:
            self.handle_dropped_files(fnames)

    def handle_dropped_files(self, file_paths):
        self.file_paths.extend(file_paths)
        self.log(f"Added {len(file_paths)} files to queue.")
        # 첫 번째 이미지 미리보기
        if file_paths:
            pixmap = QPixmap(file_paths[0])
            self.main_preview_label.setPixmap(pixmap.scaled(
                self.main_preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.main_preview_label.setText("")

    def start_processing(self):
        # 기존 로직 연결 (Controller 등)
        self.log("Starting batch processing...")
        # self.controller = ProcessingController(...) 
        # self.controller.start_processing()

    def stop_processing(self):
        self.log("Stopping processing...")
        # if self.controller: self.controller.stop()

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())