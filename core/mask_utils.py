import cv2
import numpy as np
from typing import Tuple, List, Optional

class MaskUtils:
    @staticmethod
    def refine_mask(mask: np.ndarray, dilation: int = 4, blur: int = 4) -> np.ndarray:
        """
        [ADetailer Logic] 마스크 후가공 (Seam 제거용)
        1. Dilation/Erosion: 마스크 영역 확장/축소
        2. Gaussian Blur: 경계선 부드럽게 처리
        """
        if mask is None:
            return None
            
        refined = mask.copy()

        # 1. Dilation / Erosion
        if dilation != 0:
            iterations = abs(dilation)
            kernel = np.ones((3, 3), np.uint8)
            
            if dilation > 0:
                refined = cv2.dilate(refined, kernel, iterations=iterations)
            else:
                refined = cv2.erode(refined, kernel, iterations=iterations)

        # 2. Gaussian Blur
        if blur > 0:
            # 커널 크기는 반드시 홀수 (odd)
            k_size = blur if blur % 2 == 1 else blur + 1
            refined = cv2.GaussianBlur(refined, (k_size, k_size), 0)

        return refined

    @staticmethod
    def box_to_mask(box: list, image_shape: Tuple[int, int], padding: int = 0) -> np.ndarray:
        """
        BBox 좌표를 전체 이미지 크기의 Binary Mask로 변환
        box: [x1, y1, x2, y2]
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = map(int, box)
        
        # Padding 적용 및 이미지 경계 검사 (Clamping)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def merge_masks(base_mask: Optional[np.ndarray], new_mask: np.ndarray, mode: str = "Merge") -> np.ndarray:
        """
        [Missing Logic Added] 기존 마스크와 새 마스크를 병합
        mode: "None", "Merge", "Merge and Invert"
        """
        # Base가 없으면 새 마스크를 Base로 간주
        if base_mask is None:
            result = new_mask.copy()
        else:
            # 기본적으로 합집합(Union) 처리
            result = cv2.bitwise_or(base_mask, new_mask)

        # 모드에 따른 후처리
        if mode == "Merge and Invert":
            # 합친 후 반전 (배경 인페인팅 등에 사용)
            return cv2.bitwise_not(result)
        
        # "Merge" or "None" (단순 합집합 반환)
        return result

    @staticmethod
    def combine_multiple_masks(masks: List[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        [Helper] 여러 개의 마스크 리스트를 하나로 합침 (배치 처리용)
        """
        h, w = image_shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        
        if not masks:
            return combined

        for m in masks:
            if m is not None:
                combined = cv2.bitwise_or(combined, m)
        
        return combined

    @staticmethod
    def crop_image_by_mask(image: np.ndarray, mask: np.ndarray, context_padding: int = 32) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        [BMAB Logic] 마스크 영역 기준 크롭 (Context 포함)
        Return: (cropped_image, (x1, y1, x2, y2))
        """
        if mask is None or np.max(mask) == 0:
            # 마스크가 없으면 이미지 전체 반환 (혹은 예외처리)
            return image, (0, 0, image.shape[1], image.shape[0])

        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
             return image, (0, 0, image.shape[1], image.shape[0])

        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        h, w = image.shape[:2]
        
        # Context Padding
        y_min = max(0, y_min - context_padding)
        y_max = min(h, y_max + context_padding)
        x_min = max(0, x_min - context_padding)
        x_max = min(w, x_max + context_padding)

        cropped_img = image[y_min:y_max, x_min:x_max]
        return cropped_img, (x_min, y_min, x_max, y_max)