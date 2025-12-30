import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTextEdit, QRadioButton, QSlider, QCheckBox, QGroupBox,
    QLineEdit, QSplitter, QListWidget, QListWidgetItem, QAbstractItemView,
    QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint
from PyQt6.QtGui import (
    QImage, QPixmap, QIcon, QDragEnterEvent, QDropEvent,
    QPainter, QPen, QColor, QFont
)

# =========================================================
# 1. 기본 이미지 뷰어 (메인 뷰어 및 실시간 뷰어용)
# =========================================================
class ImageCanvas(QLabel):
    point_clicked = pyqtSignal(int, int, bool)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.setMinimumSize(400, 400) 
        self.original_pixmap = None
        self.image = None

    def set_image(self, image):
        if image is None: return

        # [CRITICAL FIX] 흑백/채널 불일치로 인한 크래시 방지
        # 메모리 연속성 확보
        image = np.ascontiguousarray(image, dtype=np.uint8)
        
        # 차원 확인 (H, W)인 경우 -> (H, W, 3)으로 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1

        # OpenCV BGR/BGRA -> Qt RGB 변환
        if c == 1:
             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif c == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif c == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
        self.image = image # 변환된 이미지 저장
        
        # QImage 생성 (bytes_per_line 명시 필수)
        h, w, c = image.shape
        bytes_per_line = c * w
        q_img = QImage(
            image.data, 
            w, h, 
            bytes_per_line, 
            QImage.Format.Format_RGB888
        )
        
        # 데이터 복사로 참조 유지 (QImage가 소멸되어도 안전하게)
        self.original_pixmap = QPixmap.fromImage(q_img.copy())
        self.update_view()

    def update_view(self):
        if self.original_pixmap:
            self.setPixmap(self.original_pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        self.update_view()
        super().resizeEvent(event)

    def set_result(self, image):
        self.set_image(image)

    def get_image(self):
        return self.image

    def mousePressEvent(self, event):
        if self.original_pixmap:
            x = event.pos().x()
            y = event.pos().y()
            is_left = event.button() == Qt.MouseButton.LeftButton
            self.point_clicked.emit(x, y, is_left)


# =========================================================
# 2. [녹색 박스] 오버레이 비교 슬라이더 (Before/After Swipe)
# =========================================================
class ComparisonViewer(QWidget):
    """
    이미지를 겹쳐놓고 마우스로 긁어서 비교하는 오버레이 슬라이더 위젯
    """
    def __init__(self):
        super().__init__()
        self.pixmap_before = None
        self.pixmap_after = None
        self.slider_pos = 0.5 
        self.is_dragging = False
        
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.setMinimumSize(300, 300)

    def set_images(self, src_img, dst_img):
        """이미지 설정 및 화면 갱신"""
        self.pixmap_before = self._np2pix(src_img)
        self.pixmap_after = self._np2pix(dst_img)
        self.update() 

    def _np2pix(self, img):
        if img is None: return None
        
        # [CRITICAL FIX] 안전한 변환 로직 적용
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        if len(img.shape) == 2: # 흑백 이미지
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        
        if c == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif c == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif c == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
        # RGB로 통일됨 (c=3)
        h, w, c = img.shape
        bytes_per_line = c * w
        
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 1. 이미지가 없을 때
        if not self.pixmap_before or not self.pixmap_after:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "이미지 없음 (Before / After)")
            return

        # 2. 이미지 스케일링
        scaled_before = self.pixmap_before.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled_after = self.pixmap_after.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        # 중앙 정렬 좌표 계산
        img_w = scaled_before.width()
        img_h = scaled_before.height()
        x_offset = (w - img_w) // 2
        y_offset = (h - img_h) // 2
        
        # 3. [After 이미지] 전체 그리기 (배경)
        painter.drawPixmap(x_offset, y_offset, scaled_after)

        # 4. [Before 이미지] 클리핑하여 그리기
        split_x = int(w * self.slider_pos)
        
        painter.save()
        painter.setClipRect(0, 0, split_x, h)
        painter.drawPixmap(x_offset, y_offset, scaled_before)
        painter.restore()

        # 5. 슬라이더 라인 및 핸들
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, h)
        
        painter.setBrush(QColor(0, 120, 215)) 
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(split_x, h // 2), 6, 6)

        # 6. 텍스트 라벨
        painter.setPen(QColor(255, 255, 255))
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        
        def draw_text_with_shadow(x, y, text):
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(x+1, y+1, text)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x, y, text)

        if self.slider_pos > 0.1:
            draw_text_with_shadow(x_offset + 10, h - y_offset - 10, "Before")
        
        if self.slider_pos < 0.9:
            text_w = painter.fontMetrics().horizontalAdvance("After")
            draw_text_with_shadow(x_offset + img_w - text_w - 10, h - y_offset - 10, "After")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.update_slider_pos(event.pos().x())

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.update_slider_pos(event.pos().x())

    def mouseReleaseEvent(self, event):
        self.is_dragging = False

    def update_slider_pos(self, mouse_x):
        self.slider_pos = max(0.0, min(1.0, mouse_x / self.width()))
        self.update()


# =========================================================
# 3. [빨간 박스] 멀티 파일 업로드 대기열 (드래그앤드롭)
# =========================================================
class FileQueueWidget(QWidget):
    file_clicked = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(5)
        
        toolbar = QHBoxLayout()
        self.btn_add = QPushButton("파일 추가")
        self.btn_del_sel = QPushButton("선택 삭제")
        self.btn_del_all = QPushButton("전체 삭제")
        
        self.btn_add.setStyleSheet("background-color: #444; color: white;")
        self.btn_del_sel.setStyleSheet("background-color: #555; color: white;")
        self.btn_del_all.setStyleSheet("background-color: #d9534f; color: white;")
        
        toolbar.addWidget(self.btn_add)
        toolbar.addWidget(self.btn_del_sel)
        toolbar.addWidget(self.btn_del_all)
        layout.addLayout(toolbar)
        
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(80, 80))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setSpacing(5)
        self.list_widget.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444;")
        
        self.setAcceptDrops(True)
        self.list_widget.setAcceptDrops(False)
        self.list_widget.setDragEnabled(False)
        
        layout.addWidget(self.list_widget)
        
        self.list_widget.itemClicked.connect(self._on_click)
        self.btn_add.clicked.connect(self._open_file_dialog)
        self.btn_del_all.clicked.connect(self.list_widget.clear)
        self.btn_del_sel.clicked.connect(self._delete_selected)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                self._add_item(file_path)

    def _open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "이미지 파일 선택", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if files:
            for f in files:
                self._add_item(f)

    def _add_item(self, path):
        # 중복 체크
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == path:
                return

        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setIcon(QIcon(path))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setCheckState(Qt.CheckState.Unchecked)
        self.list_widget.addItem(item)

    def _delete_selected(self):
        items_to_remove = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked or item.isSelected():
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.list_widget.takeItem(self.list_widget.row(item))

    def _on_click(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.file_clicked.emit(path)

    def get_all_files(self):
        return [self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list_widget.count())]


# =========================================================
# 4. 로그 콘솔 (기존 유지)
# =========================================================
class LogConsole(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("background-color: #111; color: #0f0; font-family: Consolas;")
        self.setMaximumHeight(200)

    def append_log(self, message):
        self.append(message)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())