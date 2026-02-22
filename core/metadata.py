"""
core/metadata.py
이미지 저장 및 메타데이터 관리 모듈
- OpenCV가 아닌 PIL을 사용하여 PNG Info / EXIF 보존
- ADetailer 호환 포맷으로 처리 로그 기록
"""

import os
import cv2
import numpy as np
from PIL import Image, PngImagePlugin
from core.io_utils import imwrite



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

def save_image_with_metadata(cv2_image, original_path, save_path):

    """
    OpenCV 이미지(결과물)를 받아 PIL로 변환 후,
    원본의 EXIF/ICC/XMP 등을 그대로 보존하여 저장합니다.
    (더 이상 ADetailer 파라미터나 Software 태그를 추가하지 않습니다)
    """
    try:
        # 1. OpenCV(BGR) -> PIL(RGB) 변환
        if isinstance(cv2_image, np.ndarray):
            color_converted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
        else:
            pil_image = cv2_image


        # 2. 메타데이터 준비 (PNG Info)
        metadata = PngImagePlugin.PngInfo()
        
        # 3. 원본 메타데이터/ICC 프로필 보존
        exif_bytes = b""
        icc_profile = None
        info_to_preserve = {}
        
        try:
            if os.path.exists(original_path):
                original_pil = Image.open(original_path)
                
                # A. ICC Profile Preservation
                icc_profile = original_pil.info.get("icc_profile")
                
                # B. PNG Info preservation (excluding parameters and internal tags)
                # Note: We used to merge parameters here, now we just keep original if present
                existing_params = original_pil.info.get("parameters", "")
                if existing_params:
                     metadata.add_text("parameters", existing_params)
                
                # C. XMP and other metadata chunks preservation
                xmp_data = original_pil.info.get("XML:com.adobe.xmp") or original_pil.info.get("xmp")
                if xmp_data:
                    info_to_preserve["XML:com.adobe.xmp"] = xmp_data

                for k, v in original_pil.info.items():
                    if k in ["exif", "parameters", "icc_profile", "XML:com.adobe.xmp", "xmp"]:
                        continue
                    if isinstance(k, str) and (isinstance(v, str) or isinstance(v, bytes)):
                        # metadata.add_text(str(k), str(v)) # Handled in the final loop for consistency
                        info_to_preserve[k] = v


                # D. EXIF Preservation
                raw_exif = original_pil.info.get("exif")
                if not raw_exif and hasattr(original_pil, 'getexif'):
                    try:
                        raw_exif = original_pil.getexif().tobytes()
                    except Exception:
                        pass


                if raw_exif:
                    # We NO LONGER update Software or UserComment here.
                    # Just keep the raw bytes.
                    exif_bytes = raw_exif


        except Exception as e:
            print(f"[Metadata] Failed to process original metadata: {e}")

        # 4. Preserve other info markers (XMP etc)
        for k, v in info_to_preserve.items():
            try:
                if isinstance(v, bytes):
                    metadata.add_text(str(k), v.decode('latin-1'), zip=True)
                else:
                    metadata.add_text(str(k), str(v), zip=True)
            except Exception:
                pass

        
        # 5. 저장 실행
        pil_image.save(save_path, pnginfo=metadata, exif=exif_bytes, icc_profile=icc_profile, quality=95)
            
        return True

    except Exception as e:
        print(f"[Metadata] 저장 실패 ({save_path}): {e}")
        # 실패 시 비상용으로 OpenCV 저장 시도 (메타데이터는 포기)
        imwrite(save_path, cv2_image)

        return False
