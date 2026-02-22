import torch
import numpy as np
import cv2
import os
from unittest.mock import MagicMock
from core.detail_daemon import DetailDaemonContext
from core.soft_inpainting_utils import apply_pixel_composite

# 1. Test Detail Daemon Sigma Stacking
def test_dd_sigma_stacking():
    print("--- Testing Detail Daemon Sigma Stacking ---")
    
    class MockScheduler:
        def __init__(self):
            # Sigmas as a list/array that can be indexed and item() called
            self.timesteps = torch.tensor([900, 800], dtype=torch.long)
            self.sigmas = torch.tensor([10.0, 5.0, 0.0], dtype=torch.float32)
            self.step_called = 0
            
        def scale_model_input(self, sample, timestep):
            # The context manager patches this, but we need the hook to call original
            return sample 
            
        def step(self, model_output, timestep, sample, **kwargs):
            self.step_called += 1
            return MagicMock(prev_sample=sample)

    class MockPipeline:
        def __init__(self):
            self.scheduler = MockScheduler()
            self.guidance_scale = 7.5

    pipe = MockPipeline()
    config = {
        'mode': 'both',
        'amount': 0.1, 
        'start': 0.0,
        'end': 1.0,
        'bias': 0.5,
        'exponent': 1.0,
        'start_offset': 0.5,
        'end_offset': 0,
        'fade': 0,
        'smooth': False
    }
    
    with DetailDaemonContext(pipe, True, config) as dd:
        t = pipe.scheduler.timesteps[0]
        sample = torch.randn(1, 4, 64, 64)
        
        # FIRST CALL (e.g. Cond)
        _ = pipe.scheduler.scale_model_input(sample, t)
        sigma_1 = pipe.scheduler.sigmas[0].item()
        print(f"  Sigma after 1st call: {sigma_1:.4f}")
        
        # SECOND CALL (e.g. Uncond) - SHOULD NOT CHANGE SIGMA AGAIN
        _ = pipe.scheduler.scale_model_input(sample, t)
        sigma_2 = pipe.scheduler.sigmas[0].item()
        print(f"  Sigma after 2nd call: {sigma_2:.4f}")
        
        if abs(sigma_1 - sigma_2) < 1e-6:
            print("  SUCCESS: Sigma did NOT compound.")
        else:
            print(f"  FAILURE: Sigma compounded! ({sigma_1} vs {sigma_2})")
            return False

        # STEP - SHOULD RESTORE SIGMA
        pipe.scheduler.step(sample, t, sample)
        sigma_restored = pipe.scheduler.sigmas[0].item()
        print(f"  Sigma after step (restored): {sigma_restored}")
        
        if sigma_restored == 10.0:
            print("  SUCCESS: Sigma restored correctly.")
        else:
            print(f"  FAILURE: Sigma NOT restored correctly. ({sigma_restored})")
            return False
            
    return True

# 2. Test Soft Inpainting Mask Influence
def test_soft_inpainting_influence():
    print("\n--- Testing Soft Inpainting Mask Influence ---")
    
    # Create test data
    h, w = 100, 100
    orig = np.zeros((h, w, 3), dtype=np.uint8) + 100 # Dark gray
    res = np.zeros((h, w, 3), dtype=np.uint8) + 200  # Light gray
    
    # Mask: Gradient from 0 to 255
    mask = np.linspace(0, 255, w).astype(np.uint8)
    mask = np.tile(mask, (h, 1))
    
    # Config 1: Influence 0 (Standard)
    conf_0 = {'soft_mask_influence': 0.0, 'soft_diff_threshold': 0.1, 'soft_diff_contrast': 10.0}
    out_0 = apply_pixel_composite(orig, res, mask, conf_0)
    
    # Config 2: Influence 1.0 (Heavy preservation where mask is high)
    conf_1 = {'soft_mask_influence': 1.0, 'soft_diff_threshold': 0.1, 'soft_diff_contrast': 10.0}
    out_1 = apply_pixel_composite(orig, res, mask, conf_1)
    
    avg_right_0 = np.mean(out_0[:, -10:])
    avg_right_1 = np.mean(out_1[:, -10:])
    
    print(f"  Influence 0.0 - Right Avg: {avg_right_0:.1f} (Should be ~200)")
    print(f"  Influence 1.0 - Right Avg: {avg_right_1:.1f} (Should be < 200)")
    
    if avg_right_1 < avg_right_0 - 10:
        print("  SUCCESS: Mask Influence correctly biased towards Original.")
    else:
        print("  FAILURE: Mask Influence had no effect or reversed effect.")
        return False
    
    return True

if __name__ == "__main__":
    s1 = test_dd_sigma_stacking()
    s2 = test_soft_inpainting_influence()
    
    if s1 and s2:
        print("\nALL ADVANCED TESTS PASSED.")
    else:
        print("\nTESTS FAILED.")
        exit(1)
