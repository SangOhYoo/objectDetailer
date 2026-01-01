import sys
import logging
import warnings
import multiprocessing
# [Fix] Ultralytics(YOLO)와 Diffusers(Accelerate) 충돌 방지 (Meta Tensor 오류 해결)
import torch
import ultralytics

# [Fix] accelerate와 YOLO 간의 충돌(Meta Tensor 오류)을 막기 위해 init_empty_weights 패치
try:
    import accelerate
    import contextlib

    # 원래 기능을 무력화하는 더미 컨텍스트 매니저
    @contextlib.contextmanager
    def _dummy_init_empty_weights(*args, **kwargs):
        yield

    # accelerate의 함수를 덮어씌움
    accelerate.init_empty_weights = _dummy_init_empty_weights
except Exception:
    pass

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # 불필요한 라이브러리 경고 메시지 숨기기
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*The parameter 'pretrained' is deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*unexpected keys not found in the model.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Token indices sequence length is longer.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.pipelines.pipeline_utils")

    # Diffusers 라이브러리의 불필요한 로그 메시지 숨기기
    logging.getLogger("diffusers").setLevel(logging.ERROR)

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
    exit_code = app.exec()
    print(f"\n[INFO] 메인 윈도우가 닫혔습니다. 프로그램을 종료합니다. (Exit Code: {exit_code})")
    sys.exit(exit_code)