import cv2
import numpy as np
import os

def imread(path):
    """한글 경로 이미지 읽기"""
    try:
        stream = open(path.encode("utf-8"), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def imwrite(path, img):
    """한글 경로 이미지 저장"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ret, assets = cv2.imencode(os.path.splitext(path)[1], img)
        if ret:
            with open(path, mode='w+b') as f:
                assets.tofile(f)
        return ret
    except Exception as e:
        print(f"Error writing image: {e}")
        return False