import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.pipeline import ImageProcessor

class MockDetector:
    def detect(self, image, model, conf, classes=None):
        # Return a mock person detection in the center
        h, w = image.shape[:2]
        return [{
            'box': [w//3, h//3, w*2//3, h*2//3],
            'conf': 0.9,
            'label': 'person',
            'mask': None # Force fallback to box-based mask
        }]

    def offload_models(self): pass

class MockPipeline(ImageProcessor):
    def __init__(self):
        self.device = 'cpu'
        self.log_callback = print
        self.detector = MockDetector()
        self.model_manager = MagicMock()
        self.face_restorer = MagicMock()
        self.upscaler = MagicMock()

    def _run_inpaint(self, image, mask, config, denoise, box, kps=None, steps=None, guidance_scale=None):
        # Mock inpaint: Just fill mask with green (bg) or blue (harmonization)
        # If mask is full image (harmonization), tint blue
        if np.all(mask == 255):
             # Harmonization
             return cv2.addWeighted(image, 0.8, np.full_like(image, (255, 0, 0)), 0.2, 0)
        
        # Verify Canny Config
        if config.get('control_module') == 'canny':
             print("    [Mock] Verified ControlModule == 'canny'")
        else:
             print(f"    [Mock] ControlModule is {config.get('control_module')}")

        # Background Gen: fill with constant color
        out = image.copy()
        out[mask > 0] = (0, 255, 0) # Green Background
        return out

def test_advanced_composition_logic():
    # Setup
    pipeline = MockPipeline()
    img = np.zeros((512, 512, 3), dtype=np.uint8) + 100 # Gray image
    
    # Config
    config = {
        'detector_model': 'yolov8n.pt',
        'conf_thresh': 0.5,
        'resize_ratio': 0.3, # Strong zoom out
        'resize_align': 'Center',
        'resize_enable': True
    }
    
    # Run
    # We call _resize_by_person directly
    new_img, scale, sx, sy = pipeline._resize_by_person(img, config)
    
    # Verification
    # 1. Scale check
    # Original person height (mock) = 512 * (2/3 - 1/3) = 170 px.
    # Image height = 512. Ratio = 0.33.
    # Target Ratio = 0.3.
    # Scale = 0.3 / 0.33 ~ 0.9.
    
    print(f"Scale: {scale}")
    assert scale < 1.0
    
    # 2. Image Structure (Regeneration Logic)
    # Strategy: Mask should be 255 (Generate) everywhere EXCEPT the person.
    # Person center should be Protected (0) -> Mock Inpaint returns image (100).
    # Background should be Generated (255) -> Mock Inpaint returns Green [0, 255, 0].
    
    # Check Center (Person) -> Protected -> Original Gray (100)
    # BUT Global Harmonization (0.15) runs at end, potentially tinting it.
    # Mock Inpaint adds +30 to Blue channel if denoise < 1.0.
    # So expected is [100, 100, 130] (if BGR).
    # We verify it is NOT Green (0, 255, 0).
    center_px = new_img[256, 256]
    print(f"Center Pixel: {center_px}")
    assert center_px[1] < 200 # Green channel should not be blown out
    assert center_px[0] > 50 # Blue/Red should exist
    
    # Check Corner (Background) -> Generated -> Green [0, 255, 0]
    # Plus Harmonization (+30 Blue) -> [30, 255, 30]?
    corner_px = new_img[5, 5]
    print(f"Corner Pixel: {corner_px}")
    assert corner_px[1] > 200 # Green should be high
    
    # 3. Blending / Erosion Check
    # Blend region should be Green-ish (Generated)
    blend_px = new_img[35, 35] 
    print(f"Blend Region Pixel (should be Green-ish): {blend_px}")
    assert blend_px[1] > 200
    
    assert new_img.shape == img.shape
    print("Test Passed!")

if __name__ == "__main__":
    test_advanced_composition_logic()
