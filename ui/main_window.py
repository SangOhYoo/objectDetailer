import sys
import os

# [CRITICAL FIX] 프로젝트 루트 경로를 시스템 경로에 추가 (ModuleNotFoundError 해결)
# 현재 파일(ui/main_window.py)의 상위 폴더(루트)를 찾아 sys.path에 등록
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import yaml
import time
import cv2
import queue
import torch 

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QGroupBox, QLabel, QPushButton, QProgressBar, QFileDialog, 
    QTabWidget, QScrollArea, QCheckBox, QRadioButton, QComboBox, 
    QPlainTextEdit, QGridLayout, QSlider, QApplication, QFormLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap

# UI 컴포넌트 및 워커 임포트
from ui.components import ImageCanvas, LogConsole, ComparisonViewer, FileQueueWidget
from ui.workers import InitWorker, ProcessWorker
from core.io_utils import imread, imwrite
from core.logger import setup_logger

# YOLO 모델 경로 설정
YOLO_MODEL_DIR = r"D:\AI_Models\adetailer"

class PassSettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QVBoxLayout(content)

        # 1. 활성화 및 모드 설정
        self.gb_mode = QGroupBox("모델 및 모드 설정")
        l_mode = QVBoxLayout(self.gb_mode)
        
        self.chk_enable = QCheckBox("이 탭 활성화 (Enable Pass)")
        self.chk_enable.setChecked(False)
        
        row_algo = QHBoxLayout()
        self.rb_yolo = QRadioButton("YOLO (객체 감지)")
        self.rb_sam = QRadioButton("SAM3 (세그먼트)")
        self.rb_yolo.setChecked(True)
        row_algo.addWidget(self.rb_yolo)
        row_algo.addWidget(self.rb_sam)
        
        self.combo_model = QComboBox()
        self.load_models()
        
        l_mode.addWidget(self.chk_enable)
        l_mode.addLayout(row_algo)
        l_mode.addWidget(self.combo_model)
        form.addWidget(self.gb_mode)

        # 2. 프롬프트 설정
        self.gb_prompt = QGroupBox("인페인팅 프롬프트")
        l_prompt = QVBoxLayout(self.gb_prompt)
        
        self.txt_pos = QPlainTextEdit()
        self.txt_pos.setPlaceholderText("Positive Prompt")
        self.txt_pos.setMaximumHeight(60)
        
        # [기본값 고정]
        default_pos = (
            "[<lora:JooMiPark:1.1>:<lora:JooMeePark-10:0.2>], pale skin, (glossy skin:1.1), "
            "looking at another, (seductive smile:0.8), (angry:0.7), "
            "(full-face blush, embarrassed, aroused, moaning, ecstasy:1.6), (shocked:1.2)"
        )
        self.txt_pos.setPlainText(default_pos)
        
        self.txt_neg = QPlainTextEdit()
        self.txt_neg.setPlaceholderText("Negative Prompt")
        self.txt_neg.setMaximumHeight(40)
        
        # [기본값 고정]
        default_neg = (
            "(worst quality, low quality, normal quality, lowres, bokeh, blur:2.0), "
            "(polydactyly, polydactylism:2.0)"
        )
        self.txt_neg.setPlainText(default_neg)
        
        l_prompt.addWidget(self.txt_pos)
        l_prompt.addWidget(self.txt_neg)
        form.addWidget(self.gb_prompt)

        # 3. 감지 설정
        self.gb_det = QGroupBox("감지 설정 (Detection)")
        l_det = QGridLayout(self.gb_det)
        
        self.sl_conf = self.create_slider("신뢰도 임계값", 0.3, 0.0, 1.0)
        self.sl_mask_min = self.create_slider("마스크 최소 비율", 0.0, 0.0, 1.0)
        self.sl_mask_max = self.create_slider("마스크 최대 비율", 1.0, 0.0, 1.0)
        
        l_det.addLayout(self.sl_conf, 0, 0)
        l_det.addLayout(self.sl_mask_min, 0, 1)
        l_det.addLayout(self.sl_mask_max, 1, 0)
        form.addWidget(self.gb_det)

        # 4. 인페인팅 설정
        self.gb_inpaint = QGroupBox("인페인팅 (Inpainting)")
        l_inp = QGridLayout(self.gb_inpaint)
        
        self.sl_denoise = self.create_slider("디노이징 강도", 0.4, 0.0, 1.0)
        self.sl_blur = self.create_slider("마스크 블러", 4, 0, 64, int_mode=True)
        self.sl_padding = self.create_slider("패딩(px)", 32, 0, 200, int_mode=True)
        
        l_inp.addLayout(self.sl_denoise, 0, 0)
        l_inp.addLayout(self.sl_blur, 0, 1)
        l_inp.addLayout(self.sl_padding, 1, 0)
        
        self.chk_hires = QCheckBox("Hires Fix (고해상도 보정)")
        l_inp.addWidget(self.chk_hires, 1, 1)
        
        form.addWidget(self.gb_inpaint)
        
        # 5. ControlNet 설정
        self.gb_cnet = QGroupBox("컨트롤넷 (ControlNet)")
        l_cnet = QVBoxLayout(self.gb_cnet)
        self.sl_cnet_weight = self.create_slider("가중치 (Weight)", 1.0, 0.0, 2.0)
        l_cnet.addLayout(self.sl_cnet_weight)
        form.addWidget(self.gb_cnet)

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def load_models(self):
        if os.path.exists(YOLO_MODEL_DIR):
            models = [f for f in os.listdir(YOLO_MODEL_DIR) if f.endswith(('.pt', '.safetensors'))]
            self.combo_model.addItems(models)

    def create_slider(self, label, default, min_val, max_val, int_mode=False):
        layout = QVBoxLayout()
        lbl = QLabel(f"{label}: {default}")
        slider = QSlider(Qt.Orientation.Horizontal)
        
        if int_mode:
            slider.setRange(int(min_val), int(max_val))
            slider.setValue(int(default))
        else:
            slider.setRange(int(min_val * 100), int(max_val * 100))
            slider.setValue(int(default * 100))

        def update_label(val):
            disp_val = val if int_mode else val / 100.0
            lbl.setText(f"{label}: {disp_val}")
            
        slider.valueChanged.connect(update_label)
        layout.addWidget(lbl)
        layout.addWidget(slider)
        
        slider.get_real_value = lambda: slider.value() if int_mode else slider.value() / 100.0
        return layout

    def get_config(self):
        if not self.chk_enable.isChecked():
            return None
        return {
            'mode': 'yolo' if self.rb_yolo.isChecked() else 'sam',
            'model_path': os.path.join(YOLO_MODEL_DIR, self.combo_model.currentText()),
            'pos_prompt': self.txt_pos.toPlainText(),
            'neg_prompt': self.txt_neg.toPlainText(),
            'conf_thresh': self.sl_conf.itemAt(1).widget().get_real_value(),
            'denoise': self.sl_denoise.itemAt(1).widget().get_real_value(),
            'mask_blur': self.sl_blur.itemAt(1).widget().get_real_value(),
            'padding': self.sl_padding.itemAt(1).widget().get_real_value(),
            'hires_fix': self.chk_hires.isChecked(),
            'cnet_weight': self.sl_cnet_weight.itemAt(1).widget().get_real_value()
        }

class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        with open(config_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.task_queue = None
        self.active_workers = []
        
        self.init_ui()
        self.logger = setup_logger(self.log_signal)
        self.log_signal.connect(self.console.append_log)
        self.start_init()

    def init_ui(self):
        self.setWindowTitle("Standalone ADetailer - Ultimate Edition")
        self.resize(1600, 1600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(500) 
        
        gb_global = QGroupBox("기본 모델 설정 (Global Model Settings)")
        l_global = QFormLayout(gb_global)
        self.combo_ckpt = QComboBox(); self.combo_ckpt.addItems(["henmix_real_v6b.safetensors"])
        self.combo_vae = QComboBox(); self.combo_vae.addItems(["Automatic"])
        l_global.addRow("체크포인트:", self.combo_ckpt)
        l_global.addRow("VAE:", self.combo_vae)
        left_layout.addWidget(gb_global)

        self.tabs = QTabWidget()
        self.pass_tabs = []
        for i in range(1, 11): 
            tab = PassSettingsTab()
            if i == 1: tab.chk_enable.setChecked(True)
            self.tabs.addTab(tab, f"패스 {i}")
            self.pass_tabs.append(tab)
            
        left_layout.addWidget(self.tabs)
        main_layout.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        gb_process = QGroupBox("실시간 처리 (Real-time Detection)")
        gb_process.setStyleSheet("QGroupBox { border: 2px solid #FFD700; font-weight: bold; margin-top: 10px; } QGroupBox::title { color: #FFD700; }")
        l_proc = QVBoxLayout(gb_process)
        
        self.lbl_realtime = QLabel("대기 중...")
        self.lbl_realtime.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_realtime.setMinimumHeight(300)
        self.lbl_realtime.setStyleSheet("background-color: #000;")
        self.lbl_realtime.setScaledContents(True)
        l_proc.addWidget(self.lbl_realtime)
        
        gb_compare = QGroupBox("전후 비교 (Before / After)")
        gb_compare.setStyleSheet("QGroupBox { border: 2px solid #32CD32; font-weight: bold; margin-top: 10px; } QGroupBox::title { color: #32CD32; }")
        l_comp = QVBoxLayout(gb_compare)
        
        self.compare_viewer = ComparisonViewer()
        l_comp.addWidget(self.compare_viewer)
        
        gb_queue = QGroupBox("작업 대기열 (File Queue)")
        gb_queue.setStyleSheet("QGroupBox { border: 2px solid #FF4500; font-weight: bold; margin-top: 10px; } QGroupBox::title { color: #FF4500; }")
        l_queue = QVBoxLayout(gb_queue)
        
        self.file_queue = FileQueueWidget()
        self.file_queue.file_clicked.connect(self.load_preview_from_queue) 
        l_queue.addWidget(self.file_queue)

        ctrl_layout = QHBoxLayout()
        self.console = LogConsole()
        
        btn_layout = QVBoxLayout()
        self.btn_run_batch = QPushButton("🚀 일괄 실행 (Run Batch)")
        self.btn_run_batch.setFixedHeight(50)
        self.btn_run_batch.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; font-size: 14px;")
        
        self.btn_stop = QPushButton("■ 중지")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")

        btn_layout.addWidget(self.btn_run_batch)
        btn_layout.addWidget(self.btn_stop)
        
        ctrl_layout.addWidget(self.console, stretch=7)
        ctrl_layout.addLayout(btn_layout, stretch=3)

        right_layout.addWidget(gb_process, stretch=3)
        right_layout.addWidget(gb_compare, stretch=3)
        right_layout.addWidget(gb_queue, stretch=2)
        right_layout.addLayout(ctrl_layout, stretch=2)

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=3)

        self.btn_run_batch.clicked.connect(self.run_batch_process)
        self.btn_stop.clicked.connect(self.stop_all_workers)

    def start_init(self):
        self.init_worker = InitWorker(self.config)
        self.init_worker.log_msg.connect(self.logger.info)
        self.init_worker.finished.connect(lambda: self.on_init_finished())
        self.init_worker.start()

    def on_init_finished(self):
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        self.logger.info(f"[System] Initialized. Available GPUs: {gpu_names}")
        self.btn_run_batch.setEnabled(True)

    def load_preview_from_queue(self, path):
        img_src = imread(path)
        if img_src is None: return

        filename = os.path.basename(path)
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        result_path = os.path.join(out_dir, f"result_{filename}")
        
        img_dst = img_src
        if os.path.exists(result_path):
            loaded_result = imread(result_path)
            if loaded_result is not None:
                img_dst = loaded_result
                self.logger.info(f"결과물 로드: {result_path}")
            else:
                self.logger.warning("결과물 파일 로드 실패")
        else:
            self.lbl_realtime.setText("처리 전 (대기 중)")

        self.compare_viewer.set_images(img_src, img_dst)

    def update_realtime_view(self, vis_img):
        if vis_img is None: return
        h, w, c = vis_img.shape
        rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, c*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.lbl_realtime.setPixmap(pix.scaled(
            self.lbl_realtime.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def on_worker_result(self, result_data):
        file_path, result_img = result_data
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        save_path = os.path.join(out_dir, f"result_{filename}")
        imwrite(save_path, result_img)
        self.logger.info(f"[Save] 저장 완료: {save_path}")
        src_img = imread(file_path)
        self.compare_viewer.set_images(src_img, result_img)

    def on_worker_finished(self):
        self.active_workers = [w for w in self.active_workers if w.isRunning()]
        if not self.active_workers:
            self.logger.info("=== 모든 배치 작업이 완료되었습니다. ===")
            self.btn_run_batch.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def stop_all_workers(self):
        self.logger.warning("작업을 강제 중단합니다...")
        while not self.task_queue.empty():
            try: self.task_queue.get_nowait()
            except: break
        self.active_workers = []
        self.btn_run_batch.setEnabled(True)

    def run_batch_process(self):
        queue_files = self.file_queue.get_all_files()
        if not queue_files:
            self.logger.warning("대기열이 비어있습니다. 파일을 추가해주세요.")
            return

        self.task_queue = queue.Queue()
        for f in queue_files:
            self.task_queue.put(f)

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.logger.error("사용 가능한 GPU가 없습니다!")
            return

        self.logger.info(f"총 {len(queue_files)}개의 작업을 {gpu_count}개의 GPU로 병렬 처리합니다.")
        self.btn_run_batch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.active_workers = []

        active_configs = []
        for i, tab in enumerate(self.pass_tabs):
            cfg = tab.get_config()
            if cfg:
                self.logger.info(f"  - Pass {i+1} 활성화: {os.path.basename(cfg['model_path'])}")
                active_configs.append(cfg)
        
        if not active_configs:
            self.logger.warning("활성 패스가 없어 Pass 1을 강제 활성화합니다.")
            self.pass_tabs[0].chk_enable.setChecked(True)
            active_configs.append(self.pass_tabs[0].get_config())

        for i in range(gpu_count):
            worker = ProcessWorker(
                config=self.config,
                device_id=i,
                task_queue=self.task_queue,
                active_configs=active_configs
            )
            worker.log_msg.connect(self.logger.info)
            worker.result_ready.connect(self.on_worker_result)
            worker.intermediate_update.connect(self.update_realtime_view)
            worker.finished.connect(self.on_worker_finished)
            worker.start()
            self.active_workers.append(worker)
            self.logger.info(f"GPU-{i} 워커 실행됨.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())