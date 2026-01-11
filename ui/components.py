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
from ui.styles import ModernTheme

# =========================================================
# 1. 기본 이미지 뷰어 (메인 뷰어 및 실시간 뷰어용)
# =========================================================
class ImageCanvas(QLabel):
    point_clicked = pyqtSignal(int, int, bool)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_theme("dark")
        self.setMinimumSize(400, 400) 
        self.original_pixmap = None
        self.image = None

    def set_image(self, image):
        if image is None: return

        # 메모리 연속성 확보 (크래시 방지)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        
        # 차원 확인 (H, W)인 경우 -> (H, W, 3)으로 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1

        if c == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif c == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif c == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
        self.image = image
        
        h, w, c = image.shape
        bytes_per_line = c * w
        q_img = QImage(
            image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        
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

    def set_theme(self, mode):
        if mode == "dark":
            self.setStyleSheet(f"background-color: {ModernTheme.DARK_BG_MAIN}; border: 1px solid {ModernTheme.DARK_BORDER};")
        else:
            self.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BG_MAIN}; border: 1px solid {ModernTheme.LIGHT_BORDER};")


# =========================================================
# 2. [수정됨] ComparisonViewer (Before/After 스플리터)
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
        self.text_color = Qt.GlobalColor.white
        self.set_theme("dark")
        self.setMinimumSize(300, 300)

    def set_images(self, src_img, dst_img):
        self.pixmap_before = self._np2pix(src_img)
        self.pixmap_after = self._np2pix(dst_img)
        self.update() 

    def _np2pix(self, img):
        if img is None: return None
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        
        if c == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif c == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif c == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
        h, w, c = img.shape
        bytes_per_line = c * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # [FIX] 스타일시트(px)로 인한 pointSize -1 경고 방지
        # Painter 초기화 직후 안전한 폰트로 교체하여 모든 drawText 호출 시 에러 예방
        safe_font = QFont("Arial", 9)
        painter.setFont(safe_font)
        
        w = self.width()
        h = self.height()
        
        if not self.pixmap_before or not self.pixmap_after:
            # print("[DEBUG] No images to draw. Drawing placeholder text.")
            painter.setPen(self.text_color)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "이미지 없음 (Before / After)")
            return

        # 이미지 스케일링
        scaled_before = self.pixmap_before.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        scaled_after = self.pixmap_after.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        img_w = scaled_before.width()
        img_h = scaled_before.height()
        x_offset = (w - img_w) // 2
        y_offset = (h - img_h) // 2
        
        # After 이미지 (배경)
        painter.drawPixmap(x_offset, y_offset, scaled_after)

        # Before 이미지 (클리핑)
        split_x = int(w * self.slider_pos)
        
        painter.save()
        painter.setClipRect(0, 0, split_x, h)
        painter.drawPixmap(x_offset, y_offset, scaled_before)
        painter.restore()

        # 슬라이더 라인
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, h)
        
        painter.setBrush(QColor(0, 120, 215)) 
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPoint(split_x, h // 2), 6, 6)

        # 6. 텍스트 라벨
        painter.setPen(QColor(255, 255, 255))
        
        font = painter.font()
        font.setBold(True)
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

    def set_theme(self, mode):
        if mode == "dark":
            self.setStyleSheet(f"background-color: {ModernTheme.DARK_BG_MAIN}; border: 1px solid {ModernTheme.DARK_BORDER};")
            self.text_color = Qt.GlobalColor.white
        else:
            self.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BG_MAIN}; border: 1px solid {ModernTheme.LIGHT_BORDER};")
            self.text_color = Qt.GlobalColor.black
        self.update()

# =========================================================
# 3. [빨간 박스] 멀티 파일 업로드 대기열 (드래그앤드롭)
# =========================================================
class FileQueueWidget(QWidget):
    file_clicked = pyqtSignal(str)
    file_rotated = pyqtSignal(str, int) # path, angle

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(5)
        
        toolbar = QHBoxLayout()
        
        # [New] Rotation Controls
        self.btn_rot_ccw = QPushButton("↺")
        self.btn_rot_reset = QPushButton("Reset")
        self.btn_rot_cw = QPushButton("↻")
        
        # Style for rotation buttons
        self.btn_rot_ccw.setFixedWidth(50)
        self.btn_rot_reset.setFixedWidth(80)
        self.btn_rot_cw.setFixedWidth(50)
             
        self.btn_rot_ccw.setToolTip("왼쪽으로 90도 회전 (Counter-Clockwise)")
        self.btn_rot_cw.setToolTip("오른쪽으로 90도 회전 (Clockwise)")
        self.btn_rot_reset.setToolTip("회전 초기화 (Reset Rotation)")

        toolbar.addWidget(self.btn_rot_ccw)
        toolbar.addWidget(self.btn_rot_reset)
        toolbar.addWidget(self.btn_rot_cw)
        
        # Separator (Space)
        toolbar.addSpacing(10)

        self.btn_add = QPushButton("파일 추가")
        self.btn_del_sel = QPushButton("선택 삭제")
        self.btn_del_all = QPushButton("전체 삭제")
        
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
        
        self.setAcceptDrops(True)
        self.list_widget.setAcceptDrops(False)
        self.list_widget.setDragEnabled(False)
        
        layout.addWidget(self.list_widget)
        
        self.list_widget.itemClicked.connect(self._on_click)
        self.btn_add.clicked.connect(self._open_file_dialog)
        self.btn_del_all.clicked.connect(self.list_widget.clear)
        self.btn_del_sel.clicked.connect(self._delete_selected)
        
        # Connect Rotation
        self.btn_rot_ccw.clicked.connect(lambda: self._rotate_selection(-90))
        self.btn_rot_cw.clicked.connect(lambda: self._rotate_selection(90))
        self.btn_rot_reset.clicked.connect(self._reset_rotation)
        
        self.set_theme("dark")

    def _rotate_selection(self, angle_delta):
        items = self.list_widget.selectedItems()
        if not items: return
        
        for item in items:
            current_angle = item.data(Qt.ItemDataRole.UserRole + 1) or 0
            new_angle = (current_angle + angle_delta) % 360
            self._update_item_rotation(item, new_angle)
            
            # Emit signal for the last item (or all?) - usually Preview shows only last clicked
            # If multiple selected, we might confuse preview. Update preview if focused.
            if item == self.list_widget.currentItem():
                path = item.data(Qt.ItemDataRole.UserRole)
                self.file_rotated.emit(path, new_angle)

    def _reset_rotation(self):
        items = self.list_widget.selectedItems()
        if not items: return
        
        for item in items:
            self._update_item_rotation(item, 0)
            if item == self.list_widget.currentItem():
                path = item.data(Qt.ItemDataRole.UserRole)
                self.file_rotated.emit(path, 0)

    def _update_item_rotation(self, item, angle):
        item.setData(Qt.ItemDataRole.UserRole + 1, angle)
        
        # Update Icon
        path = item.data(Qt.ItemDataRole.UserRole)
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull() and angle != 0:
                transform = list_transform = None
                # QPixmap rotation
                from PyQt6.QtGui import QTransform
                t = QTransform().rotate(angle)
                pix = pix.transformed(t, Qt.TransformationMode.SmoothTransformation)
            
            # Scale for Icon
            # Cache handled by OS/Qt mostly, but rotating full pixmap everytime might be slow for huge lists? 
            # It's manual action, so okay.
            item.setIcon(QIcon(pix))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        added = False
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                self._add_item(file_path)
                added = True
        
        if added and self.list_widget.count() > 0:
            item = self.list_widget.item(0)
            self.list_widget.setCurrentItem(item)
            self._on_click(item)

    def _open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "이미지 파일 선택", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if files:
            for f in files:
                self._add_item(f)

    def _add_item(self, path):
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == path:
                return

        item = QListWidgetItem(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setData(Qt.ItemDataRole.UserRole + 1, 0) # Init Rotation 0
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

    def get_all_tasks(self):
        """Returns list of (path, angle) tuples"""
        tasks = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            angle = item.data(Qt.ItemDataRole.UserRole + 1) or 0
            tasks.append((path, angle))
        return tasks

    def get_rotation(self, path):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                return item.data(Qt.ItemDataRole.UserRole + 1) or 0
        return 0

    def select_item_by_path(self, path):
        """경로에 해당하는 아이템을 선택 상태로 변경 (시그널 발생 없음)"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                self.list_widget.setCurrentItem(item)
                self.list_widget.scrollToItem(item)
                break

    def set_theme(self, mode):
        if mode == "dark":
            self.list_widget.setStyleSheet(f"background-color: {ModernTheme.DARK_BG_INPUT}; border: 1px solid {ModernTheme.DARK_BORDER}; color: {ModernTheme.DARK_TEXT_MAIN};")
            self.btn_add.setStyleSheet(f"background-color: {ModernTheme.DARK_BTN_BG}; color: {ModernTheme.DARK_TEXT_MAIN}; border: 1px solid {ModernTheme.DARK_BORDER};")
            self.btn_del_sel.setStyleSheet(f"background-color: {ModernTheme.DARK_BTN_BG}; color: {ModernTheme.DARK_TEXT_MAIN}; border: 1px solid {ModernTheme.DARK_BORDER};")
            self.btn_del_all.setStyleSheet(f"background-color: #d9534f; color: white; border: 1px solid #c9302c;")
            # Rot buttons
            self.btn_rot_ccw.setStyleSheet(f"background-color: {ModernTheme.DARK_BTN_BG}; color: {ModernTheme.DARK_TEXT_MAIN}; border: 1px solid {ModernTheme.DARK_BORDER};")
            self.btn_rot_cw.setStyleSheet(f"background-color: {ModernTheme.DARK_BTN_BG}; color: {ModernTheme.DARK_TEXT_MAIN}; border: 1px solid {ModernTheme.DARK_BORDER};")
            self.btn_rot_reset.setStyleSheet(f"background-color: {ModernTheme.DARK_BTN_BG}; color: {ModernTheme.DARK_TEXT_MAIN}; border: 1px solid {ModernTheme.DARK_BORDER};")
        else:
            self.list_widget.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BG_INPUT}; border: 1px solid {ModernTheme.LIGHT_BORDER}; color: {ModernTheme.LIGHT_TEXT_MAIN};")
            self.btn_add.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BTN_BG}; color: {ModernTheme.LIGHT_TEXT_MAIN}; border: 1px solid {ModernTheme.LIGHT_BORDER};")
            self.btn_del_sel.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BTN_BG}; color: {ModernTheme.LIGHT_TEXT_MAIN}; border: 1px solid {ModernTheme.LIGHT_BORDER};")
            self.btn_del_all.setStyleSheet(f"background-color: #d9534f; color: white; border: 1px solid #d43f3a;")
            # Rot buttons
            self.btn_rot_ccw.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BTN_BG}; color: {ModernTheme.LIGHT_TEXT_MAIN}; border: 1px solid {ModernTheme.LIGHT_BORDER};")
            self.btn_rot_cw.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BTN_BG}; color: {ModernTheme.LIGHT_TEXT_MAIN}; border: 1px solid {ModernTheme.LIGHT_BORDER};")
            self.btn_rot_reset.setStyleSheet(f"background-color: {ModernTheme.LIGHT_BTN_BG}; color: {ModernTheme.LIGHT_TEXT_MAIN}; border: 1px solid {ModernTheme.LIGHT_BORDER};")

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