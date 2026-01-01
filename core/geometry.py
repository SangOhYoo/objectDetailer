"""
core/geometry.py
이미지 기하학 처리 모듈
- 1도 단위 정밀 회전 (Fine Rotation)
- 해부학적 구조 검증 (Anatomy Check)
- 심리스 합성 (Seamless Composite)
"""

import cv2
import numpy as np
import math

def get_rotation_angle(kps):
    """
    InsightFace 랜드마크(눈)를 기반으로 회전 각도 계산
    kps: [LeftEye, RightEye, Nose, LeftMouth, RightMouth]
    """
    # 왼쪽 눈(0)과 오른쪽 눈(1)
    l_eye = kps[0]
    r_eye = kps[1]
    
    # 각도 계산 (Atan2)
    dy = r_eye[1] - l_eye[1]
    dx = r_eye[0] - l_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def rotate_point(pt, angle, center):
    """
    점 pt를 center 중심으로 angle(도)만큼 회전
    """
    angle_rad = math.radians(angle)
    ox, oy = center
    px, py = pt

    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return [qx, qy]

def is_anatomically_correct(kps):
    """
    [해부학 검증] 눈, 코, 입의 상대적 위치가 올바른지 확인 (괴물 얼굴 필터링)
    입력된 kps가 회전되어 있을 수 있으므로, 눈의 각도를 계산하여 정자세로 보정한 후 검증함.
    """
    if kps is None or len(kps) < 5: return False
    
    # 1. 회전 각도 계산 및 가상 회전 (정자세로 변환)
    angle = get_rotation_angle(kps)
    center = kps[2] # 코(Nose)를 중심으로 회전
    rotated_kps = [rotate_point(p, -angle, center) for p in kps]
    
    # 2. Y좌표 검증 (정자세 기준: 눈 < 코 < 입)
    l_eye_y, r_eye_y = rotated_kps[0][1], rotated_kps[1][1]
    nose_y = rotated_kps[2][1]
    mouth_y = (rotated_kps[3][1] + rotated_kps[4][1]) / 2
    
    eyes_center_y = (l_eye_y + r_eye_y) / 2
    
    # 1. 코가 눈보다 아래에 있는가?
    check1 = nose_y > eyes_center_y
    # 2. 입이 코보다 아래에 있는가?
    check2 = mouth_y > nose_y
    
    return check1 and check2

def align_and_crop(image, bbox, kps=None, target_size=512, padding=0.25, force_rotate=False, borderMode=cv2.BORDER_REFLECT):
    """
    얼굴을 잘라내고, 필요시 회전하여 정자세(0도)로 만듭니다.
    반환값: cropped_img, M (변환행렬)
    """
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    
    # 패딩 적용한 크기
    crop_size = int(max(w, h) * (1 + padding))
    
    angle = 0.0
    if force_rotate and kps is not None:
        angle = get_rotation_angle(kps)

    # 변환 행렬 (Rotation + Translation)
    # 중심점(cx, cy)을 기준으로 angle만큼 회전하고, target_size로 스케일링
    M = cv2.getRotationMatrix2D((cx, cy), angle, target_size / crop_size)
    
    # 이동(Translation) 성분 조정: 결과 이미지의 중앙으로 오도록
    M[0, 2] += (target_size / 2) - cx
    M[1, 2] += (target_size / 2) - cy
    
    # 워핑 실행
    aligned = cv2.warpAffine(image, M, (target_size, target_size), flags=cv2.INTER_LINEAR, borderMode=borderMode)
    
    return aligned, M

def restore_and_paste(base_image, processed_crop, M, mask_blur=12):
    """
    처리된 얼굴(정자세)을 역변환(Inverse)하여 원래 각도로 돌리고 합성합니다.
    """
    h, w = base_image.shape[:2]
    crop_h, crop_w = processed_crop.shape[:2]
    
    # 1. 역행렬 계산 (Invert Affine)
    M_inv = cv2.invertAffineTransform(M)
    
    # 2. 처리된 이미지를 원본 크기 캔버스에 역회전하여 배치
    # borderMode=TRANSPARENT로 하면 배경은 투명하게 됨 (혹은 0)
    restored_patch = cv2.warpAffine(processed_crop, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # 3. 마스크 생성 (사각형 White Mask -> 역회전)
    mask = np.full((crop_h, crop_w), 255, dtype=np.uint8)
    # 가장자리 블러링을 위해 안쪽으로 살짝 줄임
    border = mask_blur
    cv2.rectangle(mask, (0, 0), (crop_w, crop_h), 0, border * 2) 
    # 역회전된 마스크 (원본 이미지 위에서의 영역)
    warped_mask = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_LINEAR)
    
    # 4. 마스크 블러링 (Soft Edge)
    warped_mask = cv2.GaussianBlur(warped_mask, (mask_blur*2+1, mask_blur*2+1), 0)
    
    # 5. 알파 블렌딩
    mask_alpha = warped_mask.astype(float) / 255.0
    if len(base_image.shape) == 3:
        mask_alpha = np.expand_dims(mask_alpha, axis=2)
        
    final_img = base_image.astype(float) * (1.0 - mask_alpha) + restored_patch.astype(float) * mask_alpha
    
    return np.clip(final_img, 0, 255).astype(np.uint8)