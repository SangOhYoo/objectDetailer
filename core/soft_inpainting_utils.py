import numpy as np
import cv2

def apply_pixel_composite(original, result, mask, config):
    """
    [New] Soft Inpainting: Pixel Composite
    Blends the in-painted result with the original based on mask influence and difference threshold.
    """
    try:
        # Configs
        mask_infl = config.get('soft_mask_influence', 0.0)
        diff_thresh = config.get('soft_diff_threshold', 0.5)
        diff_contrast = config.get('soft_diff_contrast', 2.0)
        
        # Convert to float for math
        F_orig = original.astype(np.float32) / 255.0
        F_res = result.astype(np.float32) / 255.0
        F_mask = mask.astype(np.float32) / 255.0
        if len(F_mask.shape) == 2: F_mask = np.expand_dims(F_mask, axis=2)
        
        # Calculate Difference (RGB distance)
        diff = np.abs(F_orig - F_res)
        # Average across channels
        diff_map = np.mean(diff, axis=2, keepdims=True)
        
        # 1. Mask Influence
        # threshold = base_threshold + (mask * mask_influence)
        # F_mask is 1.0 inside, 0.0 outside.
        threshold = diff_thresh + (F_mask * mask_infl)
        
        # 2. Sigmoid Blending (Smoothed)
        # We ensure x doesn't overflow exp, and add a small transition buffer.
        x = (diff_map - threshold) * diff_contrast
        x = np.clip(x, -20, 20) # Prevent overflow in exp
        alpha_map = 1.0 / (1.0 + np.exp(-x))
        
        # 3. Alpha Map Refinement
        # Scale by F_mask to ensure we only apply composite where the mask is active
        # This prevents "halo" effects in non-masked regions if diff is high.
        alpha_map = alpha_map * F_mask
        
        # Blend: original * (1 - alpha) + result * alpha
        composite = F_orig * (1.0 - alpha_map) + F_res * alpha_map
        
        return np.clip(composite * 255, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"    [Error] Pixel Composite failed: {e}")
        return result
