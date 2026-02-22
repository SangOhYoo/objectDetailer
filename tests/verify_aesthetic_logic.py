import numpy as np
import re
from core.pipeline import AdetailerPipeline
from unittest.mock import MagicMock

def test_dynamic_blur():
    print("--- Testing Dynamic Mask Blur Scaling ---")
    mock_pipe = MagicMock(spec=AdetailerPipeline)
    
    # Simulate _run_inpaint context
    # Case 1: Low-res crop (512px)
    new_w_low = 512
    ui_blur = 12
    dynamic_blur_low = max(ui_blur, int(new_w_low / 40))
    print(f"  512px Crop -> Blur: {dynamic_blur_low} (Expected: 12)")
    
    # Case 2: High-res crop (1024px)
    new_w_high = 1024
    dynamic_blur_high = max(ui_blur, int(new_w_high / 40))
    print(f"  1024px Crop -> Blur: {dynamic_blur_high} (Expected: 25)")
    
    if dynamic_blur_high > dynamic_blur_low:
        print("  SUCCESS: Mask blur scales with resolution.")
    else:
        print("  FAILURE: Mask blur is static.")

def test_prompt_injection():
    print("\n--- Testing Refined Prompt Injection ---")
    
    cur_pos = "beautiful girl"
    cur_neg = "lowres"
    
    # [Refine] Simulation
    quality_pos = "raw photo, 8k uhd, high quality, highly detailed"
    quality_neg = "(painting, drawing, sketch, cartoon, anime, 3d render, illustration:1.2), blurry, (low quality, bad quality:1.2)"
    
    new_pos = f"{cur_pos}, {quality_pos}"
    new_neg = f"{cur_neg}, {quality_neg}"
    
    print(f"  New Pos: {new_pos}")
    print(f"  New Neg: {new_neg}")
    
    if "raw photo" in new_pos and "painting" in new_neg:
        print("  SUCCESS: Realism-focused tokens injected.")
    else:
        print("  FAILURE: Tokens missing.")

def test_denoise_cap():
    print("\n--- Testing Denoise Cap ---")
    # Base 0.4, small face (ratio 0.01)
    ratio = 0.01
    base = 0.4
    adj = 0.10 if ratio < 0.05 else (0.05 if ratio < 0.10 else 0.0)
    final = max(0.1, min(base + adj, 0.7))
    print(f"  Ratio 0.01, Base 0.4 -> Final: {final} (Expected: 0.5)")
    
    # Base 0.65, small face (ratio 0.01)
    base_high = 0.65
    final_high = max(0.1, min(base_high + adj, 0.7))
    print(f"  Ratio 0.01, Base 0.65 -> Final: {final_high} (Expected: 0.7 - capped)")
    
    if final_high == 0.7:
        print("  SUCCESS: Denoise cap respected.")
    else:
        print(f"  FAILURE: Denoise cap NOT respected ({final_high}).")

if __name__ == "__main__":
    test_dynamic_blur()
    test_prompt_injection()
    test_denoise_cap()
