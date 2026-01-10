
import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.bmab_utils import apply_bmab_basic, calc_color_temperature

def test_calc_color_temperature():
    # Test 0 (Neutral)
    r, g, b = calc_color_temperature(0)
    print(f"Temp 0: R={r:.2f}, G={g:.2f}, B={b:.2f}")
    assert 0.9 <= r <= 1.1
    assert 0.9 <= g <= 1.1
    assert 0.9 <= b <= 1.1

    # Test +2000 (Cool/Blue - 8500K)
    r, g, b = calc_color_temperature(2000)
    print(f"Temp +2000: R={r:.2f}, G={g:.2f}, B={b:.2f}")
    # Expect Blue > Red (Higher Kelvin is clearer/bluer)
    assert b > r

    # Test -2000 (Warm/Red - 4500K)
    r, g, b = calc_color_temperature(-2000)
    print(f"Temp -2000: R={r:.2f}, G={g:.2f}, B={b:.2f}")
    # Expect Red > Blue
    assert r > b

def test_apply_bmab_basic():
    # Create valid BGR image (100x100), mid-grey
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    
    # Config 1: No effect
    config = {
        'bmab_contrast': 1.0,
        'bmab_brightness': 1.0, 
        'bmab_sharpness': 1.0,
        'bmab_color_saturation': 1.0,
        'bmab_color_temperature': 0,
        'bmab_noise_alpha': 0.0,
        'bmab_noise_alpha_final': 0.0
    }
    
    out = apply_bmab_basic(img, config)
    assert out.shape == img.shape
    assert np.allclose(out, img, atol=1), "No effect should result in identical image"
    
    # Config 2: High Brightness
    config['bmab_brightness'] = 1.5
    out = apply_bmab_basic(img, config)
    assert np.mean(out) > np.mean(img), "Increase brightness should increase mean pixel value"
    
    # Config 3: Color Temp (Warm)
    config['bmab_brightness'] = 1.0
    config['bmab_color_temperature'] = -2000 
    out = apply_bmab_basic(img, config)
    # BGR format. Warm means Blue channel (0) should decrease (or Red increase more) relative to Red channel (2)
    # Since inputs were gray (R=G=B=128), output R should be > B
    mean_b = np.mean(out[:, :, 0])
    mean_r = np.mean(out[:, :, 2])
    print(f"Image Warm Test: Mean B={mean_b:.2f}, Mean R={mean_r:.2f}")
    assert mean_r > mean_b, "Warm temp should result in Red > Blue"

if __name__ == "__main__":
    test_calc_color_temperature()
    test_apply_bmab_basic()
    print("All tests passed!")
