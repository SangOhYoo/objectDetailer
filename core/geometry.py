import cv2
import numpy as np

def align_and_crop_with_rotation(image, bbox, landmarks, target_size=512, padding=0.5):
    """
    [성공 패턴] 랜드마크를 기반으로 얼굴 각도를 계산하여 정방향으로 회전 및 크롭합니다.
    """
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # 1. 눈 위치를 기준으로 회전 각도 계산
    # landmarks[0]: 왼쪽 눈, landmarks[1]: 오른쪽 눈
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    # 두 눈 사이의 각도 계산 (라디안 -> 도)
    angle = np.degrees(np.arctan2(dy, dx))
    
    # 2. 박스 중심 및 크기 계산
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    side = int(max(w, h) * (1 + padding))
    
    # 3. 회전이 포함된 변환 행렬 M 생성
    # 중심(cx, cy)을 기준으로 -angle만큼 회전하고 크기를 맞춤
    M = cv2.getRotationMatrix2D((cx, cy), angle, target_size / side)
    
    # 변환 행렬의 이동(Translation) 성분 조정하여 target_size 중앙에 배치
    M[0, 2] += (target_size / 2) - cx
    M[1, 2] += (target_size / 2) - cy
    
    # 4. 이미지 변환 (이제 얼굴은 정방향이 됨)
    cropped = cv2.warpAffine(image, M, (target_size, target_size), flags=cv2.INTER_LINEAR)
    
    return cropped, M

# composite_seamless 함수는 기존과 동일하게 사용합니다.
# M_inv(역행렬)가 이미 포함되어 있어, M에 회전이 들어가면 복원 시 자동으로 원상태로 돌아옵니다.


def align_and_crop(image, bbox, target_size=512, padding=0.5):
    """탐지된 박스를 기반으로 얼굴을 정렬하고 크롭합니다."""
    # 원본이 4채널(RGBA)인 경우 3채널(BGR)로 변환하여 처리
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    side = int(max(w, h) * (1 + padding))
    
    src_pts = np.array([[cx - side//2, cy - side//2], 
                        [cx + side//2, cy - side//2], 
                        [cx - side//2, cy + side//2]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [target_size, 0], [0, target_size]], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    cropped = cv2.warpAffine(image, M, (target_size, target_size), flags=cv2.INTER_LINEAR)
    
    return cropped, M

def composite_seamless(base_img, detail_img, mask, M):
    """Alpha Blending 시 채널 수를 일치시켜 ValueError를 방지합니다."""
    # [해결] base_img가 4채널(RGBA)이면 3채널(BGR)로 변환
    if base_img.shape[2] == 4:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGRA2BGR)

    mask_blur = cv2.GaussianBlur(mask, (21, 21), 11)
    mask_alpha = mask_blur.astype(float) / 255.0
    if len(mask_alpha.shape) == 2:
        mask_alpha = cv2.merge([mask_alpha, mask_alpha, mask_alpha])

    h, w = base_img.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    restored_img = cv2.warpAffine(detail_img, M_inv, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 이제 (344, 516, 3)으로 형상이 일치하여 연산이 가능합니다
    blended = base_img.astype(float) * (1.0 - mask_alpha) + restored_img.astype(float) * mask_alpha
    
    return np.clip(blended, 0, 255).astype(np.uint8)