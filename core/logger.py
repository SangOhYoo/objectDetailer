import logging
import sys
import os
from PyQt6.QtCore import QObject, pyqtSignal

class QLogHandler(logging.Handler):
    """PyQt 시그널로 로그 메시지를 전송하는 핸들러"""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        if self.signal:
            self.signal.emit(msg)

def setup_logger(signal=None):
    logger = logging.getLogger("SAM3_Detailer")
    logger.setLevel(logging.DEBUG)
    
    # 중복 핸들러 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    # [수정됨] 로그 폴더 자동 생성 로직 추가
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"[INFO] Created log directory: {os.path.abspath(log_dir)}")

    # 파일 핸들러 (debug.log)
    fh = logging.FileHandler(os.path.join(log_dir, "debug.log"), encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # 콘솔 핸들러 (CMD 창 출력)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(ch)
    
    # GUI 핸들러 (화면 출력)
    if signal:
        gh = QLogHandler(signal)
        gh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(gh)
        
    return logger