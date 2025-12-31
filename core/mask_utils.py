import cv2
import numpy as np
from typing import Tuple, List

class MaskUtils:
    @staticmethod
    def refine_mask(mask: np.ndarray, dilation: int = 4, blur: int = 4) -> np.ndarray:
        """
        [ADetailer Port] 마스크 후가공
        1. Dilation: 마스크 영역 확장 (Seam 제거)
        2. Blur: 경계선 부드럽게 처리 (자연스러운 합성)
        """
        refined = mask.copy()

        # 1. Dilation (Erosion is negative dilation)
        if dilation != 0:
            kernel_size = abs(dilation)
            kernel = np.ones((3, 3), np.uint8)
            iterations = abs(dilation)
            
            if dilation > 0:
                refined = cv2.dilate(refined, kernel, iterations=iterations)
            else:
                refined = cv2.erode(refined, kernel, iterations=iterations)

        # 2. Gaussian Blur
        if blur > 0:
            # Kernel size must be odd
            k_size = blur if blur % 2 == 1 else blur + 1
            refined = cv2.GaussianBlur(refined, (k_size, k_size), 0)

        return refined

    @staticmethod
    def box_to_mask(box: list, image_shape: Tuple[int, int], padding: int = 0) -> np.ndarray:
        """
        BBox 좌표를 Binary Mask로 변환
        box: [x1, y1, x2, y2]
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = map(int, box)
        
        # Apply Padding safely
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def crop_image_by_mask(image: np.ndarray, mask: np.ndarray, context_padding: int = 32) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        마스크 영역을 기준으로 이미지를 크롭 (Inpainting 전 단계)
        BMAB Style: 주변 컨텍스트를 포함하기 위해 padding을 적용함
        """
        if np.max(mask) == 0:
            return image, (0, 0, image.shape[1], image.shape[0])

        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        h, w = image.shape[:2]
        
        # Add Context Padding
        y_min = max(0, y_min - context_padding)
        y_max = min(h, y_max + context_padding)
        x_min = max(0, x_min - context_padding)
        x_max = min(w, x_max + context_padding)

        # Ensure width/height are multiples of 8 (Requirement for SD VAE)
        # This part is handled in the worker resizing logic usually, but good to align here if possible.

        cropped_img = image[y_min:y_max, x_min:x_max]
        return cropped_img, (x_min, y_min, x_max, y_max)

    @staticmethod
    def merge_masks(base_mask: np.ndarray, new_mask: np.ndarray, mode: str = "Merge") -> np.ndarray:
        """
        여러 객체의 마스크를 합치는 방식 정의
        """
        if base_mask is None:
            return new_mask
            
        if mode == "Merge":
            return np.maximum(base_mask, new_mask)
        elif mode == "Merge and Invert":
            # 배경을 마스킹하는 특수 케이스
            merged = np.maximum(base_mask, new_mask)
            return cv2.bitwise_not(merged)
        
        return base_mask