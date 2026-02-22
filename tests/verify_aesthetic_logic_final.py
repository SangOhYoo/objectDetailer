import numpy as np
import re

# 1. Test Dynamic Blur Scaling Logic
def test_dynamic_blur():
    print("--- Testing Dynamic Mask Blur Scaling ---")
    
    def get_dynamic_blur(ui_blur, new_w):
        return max(ui_blur, int(new_w / 40))
        
    ui_blur = 12
    # Case 1: 512px
    blur_512 = get_dynamic_blur(ui_blur, 512)
    print(f"  512px Crop -> Blur: {blur_512} (Expected: 12)")
    
    # Case 2: 1280px (Large face in high-res)
    blur_1280 = get_dynamic_blur(ui_blur, 1280)
    print(f"  1280px Crop -> Blur: {blur_1280} (Expected: 32)")
    
    if blur_1280 > blur_512:
        print("  SUCCESS: Mask blur scales with resolution.")
    else:
        print("  FAILURE: Mask blur is static.")
        return False
    return True

# 2. Test Refined Prompt Injection
def test_prompt_injection():
    print("\n--- Testing Refined Prompt Injection ---")
    
    def inject_prompts(cur_pos, cur_neg):
        quality_pos = "raw photo, 8k uhd, high quality, highly detailed"
        quality_neg = "(painting, drawing, sketch, cartoon, anime, 3d render, illustration:1.2), blurry, (low quality, bad quality:1.2)"
        
        if quality_pos not in cur_pos.lower():
            cur_pos = f"{cur_pos}, {quality_pos}" if cur_pos.strip() else quality_pos
        if quality_neg not in cur_neg.lower():
            cur_neg = f"{cur_neg}, {quality_neg}" if cur_neg.strip() else quality_neg
        return cur_pos, cur_neg

    pos, neg = inject_prompts("beautiful girl", "lowres")
    print(f"  Result Pos: {pos}")
    print(f"  Result Neg: {neg}")
    
    if "raw photo" in pos and "painting" in neg:
        print("  SUCCESS: Realism-focused tokens injected.")
    else:
        print("  FAILURE: Tokens missing.")
        return False
    return True

# 3. Test Denoise Cap
def test_denoise_cap():
    print("\n--- Testing Denoise Cap ---")
    
    def calc_dynamic_denoise(ratio, base):
        adj = 0.10 if ratio < 0.05 else (0.05 if ratio < 0.10 else 0.0)
        return max(0.1, min(base + adj, 0.7))

    # Test Case: Small face (0.01 ratio) with high base (0.65)
    final = calc_dynamic_denoise(0.01, 0.65)
    print(f"  Small Face (0.01), Base 0.65 -> Final: {final} (Expected: 0.7 - capped)")
    
    if final == 0.7:
        print("  SUCCESS: Denoise cap respected.")
    else:
        print(f"  FAILURE: Denoise cap NOT respected ({final}).")
        return False
    return True

# 4. Test Soft Inpainting Alpha Transition Safety
def test_sigmoid_clipping():
    print("\n--- Testing Sigmoid Clipping Safety ---")
    
    def get_alpha(diff, thresh, contrast):
        x = (diff - thresh) * contrast
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))

    # High contrast, high diff -> should not overflow
    alpha = get_alpha(0.9, 0.1, 100.0)
    print(f"  High Contrast (100.0) -> Alpha: {alpha}")
    
    if not np.isnan(alpha) and alpha > 0.999:
        print("  SUCCESS: Sigmoid clipped safely.")
    else:
        print("  FAILURE: Sigmoid overflow or NaN.")
        return False
    return True

if __name__ == "__main__":
    results = [
        test_dynamic_blur(),
        test_prompt_injection(),
        test_denoise_cap(),
        test_sigmoid_clipping()
    ]
    
    if all(results):
        print("\nALL AESTHETIC LOGIC TESTS PASSED.")
    else:
        print("\nTESTS FAILED.")
        exit(1)
