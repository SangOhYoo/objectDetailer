from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor

class ScheduleGraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 200)
        self.y_values = []
        self.set_theme("dark")
        
    def set_theme(self, mode):
        if mode == "dark":
            self.bg_color = QColor(30, 30, 30)
            self.grid_color = QColor(60, 60, 60)
            self.text_color = QColor(200, 200, 200)
            self.line_color = QColor(100, 180, 255) # Light Blue
        else:
            self.bg_color = QColor(250, 250, 250)
            self.grid_color = QColor(200, 200, 200)
            self.text_color = QColor(50, 50, 50)
            self.line_color = QColor(0, 120, 215) # Blue
        self.update()

    def set_data(self, values):
        """
        values: list or numpy array of float values (multipliers).
        """
        self.y_values = values
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 1. Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # 2. Grid & Axes
        # Fixed Range: Y is -1.0 to 1.0 based on screenshot
        y_min, y_max = -1.0, 1.0
        
        # Margins
        margin_left = 40
        margin_right = 10
        margin_top = 10
        margin_bottom = 20
        
        graph_w = w - margin_left - margin_right
        graph_h = h - margin_top - margin_bottom
        
        # Draw Box (Border)
        painter.setPen(QPen(self.grid_color))
        painter.drawRect(margin_left, margin_top, int(graph_w), int(graph_h))
        
        pen_grid = QPen(self.grid_color)
        pen_grid.setStyle(Qt.PenStyle.DotLine)
        
        # Horizontal Grid Lines (0.0, +/-0.5, +/-1.0)
        lines = [0.0, 0.5, 1.0, -0.5, -1.0]
        
        for val in lines:
            norm_y = (val - y_min) / (y_max - y_min) # 0..1 (0=-1, 1=1)
            screen_y = margin_top + (1.0 - norm_y) * graph_h
            
            # Grid Line
            if val == 0:
                painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.PenStyle.SolidLine))
            else:
                painter.setPen(pen_grid)
            
            if margin_top <= screen_y <= h - margin_bottom:
                painter.drawLine(int(margin_left), int(screen_y), int(w - margin_right), int(screen_y))
            
            # Label
            painter.setPen(self.text_color)
            painter.drawText(QRect(0, int(screen_y) - 10, margin_left - 5, 20), 
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{val:.2f}")

        # Vertical Center Line (0.5 steps)
        painter.setPen(pen_grid)
        center_x = margin_left + graph_w / 2
        painter.drawLine(int(center_x), margin_top, int(center_x), h - margin_bottom)

        # 3. Curve
        if len(self.y_values) > 1:
            raw_path = []
            steps = len(self.y_values)
            
            for i, val in enumerate(self.y_values):
                # X
                x = margin_left + (i / (steps - 1)) * graph_w
                
                # Y
                disp_val = max(y_min, min(y_max, val))
                norm_y = (disp_val - y_min) / (y_max - y_min)
                y = margin_top + (1.0 - norm_y) * graph_h
                
                raw_path.append(QPoint(int(x), int(y)))
                
            pen_curve = QPen(self.line_color)
            pen_curve.setWidth(2)
            painter.setPen(pen_curve)
            
            # Draw Polyline
            painter.drawPolyline(raw_path)
