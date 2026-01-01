import sys
import os
import logging
import warnings
import multiprocessing

# [Fix] TensorFlow/MediaPipe(InsightFace) C++ 로그 숨기기 (import 전에 설정)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

# [Fix] accelerate와 YOLO/GFPGAN 간의 충돌(Meta Tensor 오류)을 막기 위해 init_empty_weights 패치
# Ultralytics나 Diffusers가 로드되기 전에 패치해야 안전함
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

# [Fix] Ultralytics(YOLO)와 Diffusers(Accelerate) 충돌 방지
import torch
import ultralytics

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
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Already found a `peft_config` attribute.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.pipelines.pipeline_utils")
    
    # 라이브러리 불필요한 로그 메시지 숨기기 (Diffusers, Transformers, PEFT)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    exit_code = app.exec()
    print(f"\n[INFO] 메인 윈도우가 닫혔습니다. 프로그램을 종료합니다. (Exit Code: {exit_code})", flush=True)
    sys.exit(exit_code)