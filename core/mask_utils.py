import cv2
import numpy as np

class MaskUtils:
    @staticmethod
    def refine_mask(mask: np.ndarray, dilation: int = 4, blur: int = 8) -> np.ndarray:
        """
        [ADetailer Logic Implementation]
        Seam removal logic: Dilation -> Blur
        """
        if mask is None:
            return None
            
        refined = mask.copy()

        # 1. Dilation
        if dilation > 0:
            kernel = np.ones((3, 3), np.uint8)
            refined = cv2.dilate(refined, kernel, iterations=dilation)

        # 2. Gaussian Blur
        if blur > 0:
            k_size = blur if blur % 2 == 1 else blur + 1
            refined = cv2.GaussianBlur(refined, (k_size, k_size), 0)

        return refined

    @staticmethod
    def crop_image_by_mask(image, mask, context_padding=32):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return image, (0, 0, image.shape[1], image.shape[0])

        h, w = image.shape[:2]
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        y_min = max(0, y_min - context_padding)
        y_max = min(h, y_max + context_padding)
        x_min = max(0, x_min - context_padding)
        x_max = min(w, x_max + context_padding)

        return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)