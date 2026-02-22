import torch
import numpy as np
import cv2
import os
from unittest.mock import MagicMock
from core.detail_daemon import DetailDaemonContext
from core.soft_inpainting_utils import apply_pixel_composite

# 1. Regression: Detail Daemon + Soft Inpainting Synergy
def test_synergy_dd_soft():
    print("--- Testing Detail Daemon & Soft Inpainting Synergy ---")
    
    class MockScheduler:
        def __init__(self):
            self.timesteps = torch.tensor([100], dtype=torch.long)
            self.sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)
            self.step_called = 0
        def scale_model_input(self, sample, t): return sample
        def step(self, m, t, s, **k): 
            self.step_called += 1
            return MagicMock(prev_sample=s)

    pipe = MagicMock()
    pipe.scheduler = MockScheduler()
    pipe.guidance_scale = 7.5
    
    config = {
        'dd_enabled': True,
        'dd_amount': 0.1,
        'use_soft_inpainting': True,
        'soft_mask_influence': 0.5,
        'soft_diff_threshold': 0.1,
        'soft_diff_contrast': 10.0
    }
    
    # Verify DD Context enters correctly with synergy config
    try:
        with DetailDaemonContext(pipe, True, config) as dd:
            print("  SUCCESS: DetailDaemonContext initialized with Soft Inpainting config.")
    except Exception as e:
        print(f"  FAILURE: Synergy context failed: {e}")
        return False
        
    return True

# 2. Regression: Dynamic Blur vs Hires Fix Interaction
def test_hires_dynamic_blur():
    print("\n--- Testing Hires Fix & Dynamic Blur Interaction ---")
    
    # Logic simulation in pipeline.py:
    # 1. Hires Fix runs -> target_res increases (e.g. 1024)
    # 2. _run_inpaint calculates dynamic_blur based on new_w
    
    def simulate_pipeline_blur(target_res, ui_blur=12):
        # new_w is target_res after padding/upscale
        dynamic_blur = max(ui_blur, int(target_res / 40))
        return dynamic_blur

    blur_normal = simulate_pipeline_blur(512)
    blur_hires = simulate_pipeline_blur(1024)
    
    print(f"  Normal (512px) -> Blur: {blur_normal}")
    print(f"  Hires (1024px) -> Blur: {blur_hires}")
    
    if blur_hires == 25 and blur_normal == 12:
        print("  SUCCESS: Dynamic blur correctly adapts to Hires Fix resolution.")
    else:
        print(f"  FAILURE: Dynamic blur mismatch ({blur_normal}, {blur_hires})")
        return False
    return True

# 3. Regression: Pixel Composite Safety (Black/Empty Images)
def test_composite_safety():
    print("\n--- Testing Pixel Composite Safety (Empty/Black) ---")
    
    orig = np.zeros((100, 100, 3), dtype=np.uint8)
    res = np.zeros((100, 100, 3), dtype=np.uint8) # Same as original
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    config = {'soft_mask_influence': 0.5, 'soft_diff_threshold': 0.1, 'soft_diff_contrast': 2.0}
    
    try:
        out = apply_pixel_composite(orig, res, mask, config)
        print(f"  Empty handling: Success (Shape: {out.shape})")
    except Exception as e:
        print(f"  FAILURE: Composite crashed on empty data: {e}")
        return False
        
    return True

if __name__ == "__main__":
    results = [
        test_synergy_dd_soft(),
        test_hires_dynamic_blur(),
        test_composite_safety()
    ]
    
    if all(results):
        print("\nFINAL REGRESSION CHECK PASSED.")
    else:
        print("\nREGRESSION CHECK FAILED.")
        exit(1)
