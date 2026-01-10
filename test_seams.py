
import cv2
import numpy as np
from core.geometry import restore_and_paste

def test_seams():
    # 1. Base Image (White)
    h, w = 512, 512
    base = np.full((h, w, 3), 200, dtype=np.uint8)
    
    # 2. Processed Crop (Red)
    crop_size = 256
    processed = np.full((crop_size, crop_size, 3), (0, 0, 255), dtype=np.uint8)
    
    # 3. Mask (Circle)
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 100, 255, -1)
    
    # 4. Paste back params
    # M that places crop in center
    cx, cy = w//2, h//2
    # translate from crop center(128,128) to base center(256,256)
    # Target(256) = M * Source(128) + T
    # 1.0 * 128 + T = 256 => T = 128
    M = np.array([[1.0, 0, 128.0], [0, 1.0, 128.0]])
    
    # Case A: Current Logic (mask_blur might be ignored if paste_mask is passed)
    res = restore_and_paste(base.copy(), processed, M, mask_blur=0, paste_mask=mask)
    cv2.imwrite("test_seam_sharp.png", res)
    
    # Case B: With Blur
    # Currently pipeline passes mask_blur to restore_and_paste, but restore_and_paste line 142 says:
    # if paste_mask is None and mask_blur > 0: ...
    # This means if paste_mask IS passed (which it is in pipeline.py), blurring is SKIPPED in geometry.py!
    # This assumes paste_mask was ALREADY blurred.
    
    res_blurred = restore_and_paste(base.copy(), processed, M, mask_blur=10, paste_mask=mask)
    cv2.imwrite("test_seam_ignored_blur.png", res_blurred)
    
    print("Test images generated.")

if __name__ == "__main__":
    test_seams()
