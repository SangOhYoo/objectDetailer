"""
core/metadata.py
이미지 저장 및 메타데이터 관리 모듈
- OpenCV가 아닌 PIL을 사용하여 PNG Info / EXIF 보존
- ADetailer 호환 포맷으로 처리 로그 기록
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, PngImagePlugin, ImageOps

def load_image_as_pil(image_path):
    """
    이미지 경로에서 PIL 객체를 생성합니다 (EXIF 정보 포함).
    """
    try:
        # PIL은 Lazy Loading하므로, 메타데이터 보존을 위해 open 후 바로 copy() 권장
        img = Image.open(image_path)
        img = img.convert("RGB") # RGBA -> RGB 변환 (JPEG 호환성)
        return img
    except Exception as e:
        print(f"[Metadata] 이미지 로드 실패: {e}")
        return None

def save_image_with_metadata(cv2_image, original_path, save_path, config):
    """
    OpenCV 이미지(결과물)를 받아 PIL로 변환 후,
    원본의 EXIF를 이식하고 ADetailer 호환 로그를 심어서 저장합니다.
    """
    try:
        # 1. OpenCV(BGR) -> PIL(RGB) 변환
        if isinstance(cv2_image, np.ndarray):
            color_converted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
        else:
            pil_image = cv2_image # 이미 PIL이면 그대로 사용

        # 2. 메타데이터 준비 (PNG Info)
        metadata = PngImagePlugin.PngInfo()
        
        # 3. ADetailer 스타일의 파라미터 텍스트 생성
        # config 객체에서 JSON 변환 메서드 호출
        ad_params = config.to_adetailer_json()
        
        # 보기 좋은 텍스트 형태로 변환 (WebUI 스타일)
        # 예: "Model: face_yolo, Confidence: 0.35, Denoise: 0.4 ..."
        param_text = ", ".join([f"{k}: {v}" for k, v in ad_params.items()])
        
        # 'parameters' 키에 저장 (표준 규격)
        metadata.add_text("parameters", param_text)
        metadata.add_text("Software", "SAM3_FaceDetailer_Ultimate")
        metadata.add_text("Comment", json.dumps(ad_params)) # 기계 분석용 JSON도 별도 저장

        # 4. 원본 EXIF 이식 (선택 사항)
        # 원본 사진을 다시 열어서 exif 정보를 가져옴
        exif_bytes = None
        try:
            original_pil = Image.open(original_path)
            exif_bytes = original_pil.info.get("exif")
        except:
            pass # 원본에 EXIF가 없으면 무시

        # 5. 저장 실행
        # exif 데이터가 있으면 같이 저장
        if exif_bytes:
            pil_image.save(save_path, pnginfo=metadata, exif=exif_bytes, quality=95)
        else:
            pil_image.save(save_path, pnginfo=metadata, quality=95)
            
        return True

    except Exception as e:
        print(f"[Metadata] 저장 실패 ({save_path}): {e}")
        # 실패 시 비상용으로 OpenCV 저장 시도 (메타데이터는 포기)
        cv2.imwrite(save_path, cv2_image)
        return False