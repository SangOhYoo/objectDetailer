import sys
import warnings
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    # 불필요한 라이브러리 경고 메시지 숨기기
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*The parameter 'pretrained' is deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

    app = QApplication(sys.argv)

    # 탭 선택 가시성을 높이기 위한 스타일시트 적용
    app.setStyleSheet("""
        QTabBar::tab {
            background: #E0E0E0;
            color: #000000;
            padding: 8px 15px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #0078D7;
            color: #FFFFFF;
            font-weight: bold;
        }
        QTabBar::tab:hover:!selected {
            background: #D0D0D0;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())