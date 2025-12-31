import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QLabel, QPushButton, QSplitter, 
                             QTextEdit, QFileDialog, QProgressBar, QListWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QIcon

# Import Custom Widgets
from ui.main_window_tabs import AdetailerUnitWidget
from ui.workers import ProcessingController

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 FaceDetailer Ultimate (ADetailer UI Clone)")
        self.resize(1400, 900)
        
        self.file_paths = []
        self.controller = None
        
        self.init_ui()
        self.apply_stylesheet()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ==========================================
        # [Left Panel] ADetailer Settings (Tabs)
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        lbl_title = QLabel("ADetailer Configuration")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        left_layout.addWidget(lbl_title)

        self.tabs = QTabWidget()
        # Create 3 Units (ADetailer 1st, 2nd, 3rd)
        self.unit_widgets = []
        for i, name in enumerate(["1st", "2nd", "3rd"]):
            unit = AdetailerUnitWidget(unit_name=name)
            self.unit_widgets.append(unit)
            self.tabs.addTab(unit, f"ADetailer {name}")
        
        left_layout.addWidget(self.tabs)

        # ==========================================
        # [Center Panel] Queue & Control
        # ==========================================
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(5, 5, 5, 5)
        
        # Queue List
        self.lbl_queue = QLabel("Job Queue (0 files)")
        self.list_queue = QListWidget()
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("QProgressBar { height: 25px; }")
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Files / Folder")
        self.btn_add.clicked.connect(self.add_files)
        self.btn_add.setMinimumHeight(40)
        
        self.btn_start = QPushButton("Generate (Dual GPU)")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; font-size: 14px;")
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_start)
        
        center_layout.addWidget(self.lbl_queue)
        center_layout.addWidget(self.list_queue)
        center_layout.addWidget(self.progress_bar)
        center_layout.addLayout(btn_layout)

        # ==========================================
        # [Right Panel] Preview & Log
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        self.viewer_label = QLabel("Preview")
        self.viewer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.viewer_label.setMinimumHeight(400)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 11px;")

        right_layout.addWidget(self.viewer_label, 2)
        right_layout.addWidget(self.log_text, 1)

        # ==========================================
        # [Splitter Integration]
        # ==========================================
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([450, 300, 650]) # Initial widths
        
        main_layout.addWidget(splitter)

    def apply_stylesheet(self):
        # Dark Theme akin to WebUI
        self.setStyleSheet("""
            QMainWindow { background-color: #333; color: #eee; }
            QWidget { color: #eee; }
            QTabWidget::pane { border: 1px solid #555; }
            QTabBar::tab { background: #444; padding: 8px; margin-right: 2px; }
            QTabBar::tab:selected { background: #666; font-weight: bold; }
            QGroupBox { border: 1px solid #555; margin-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }
            QTextEdit, QListWidget { background-color: #222; border: 1px solid #555; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #444; border: 1px solid #555; padding: 4px; }
            QPushButton { background-color: #555; border: 1px solid #666; border-radius: 4px; }
            QPushButton:hover { background-color: #666; }
        """)

    def log(self, message):
        self.log_text.append(message)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def add_files(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if fnames:
            for f in fnames:
                if f not in self.file_paths:
                    self.file_paths.append(f)
                    self.list_queue.addItem(os.path.basename(f))
            self.lbl_queue.setText(f"Job Queue ({len(self.file_paths)} files)")
            self.log(f"Added {len(fnames)} files.")

    def start_processing(self):
        if not self.file_paths:
            self.log("Error: Queue is empty.")
            return

        # Gather configs from all units
        configs = [unit.get_config() for unit in self.unit_widgets]
        
        # Filter only enabled units
        active_configs = [c for c in configs if c['enabled']]
        
        if not active_configs:
            self.log("Error: Enable at least one ADetailer Unit.")
            return

        self.btn_add.setEnabled(False)
        self.btn_start.setEnabled(False)
        
        # Use the robust Controller from previous step
        self.controller = ProcessingController(self.file_paths, active_configs)
        self.controller.log_signal.connect(self.log)
        self.controller.progress_signal.connect(self.update_progress)
        self.controller.result_signal.connect(self.display_result)
        self.controller.finished_signal.connect(self.on_finished)
        
        self.controller.start_processing()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Processing... {current}/{total}")

    def display_result(self, filepath, img_array):
        import cv2
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.viewer_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.viewer_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def on_finished(self):
        self.log("=== All Tasks Finished ===")
        self.btn_add.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.progress_bar.setFormat("Done")

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())