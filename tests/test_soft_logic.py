import unittest
import numpy as np
import cv2
import sys
import os
from unittest.mock import MagicMock

# [Mock] Mock diffusers to avoid DLL/Import errors during logic test
sys.modules['diffusers'] = MagicMock()
sys.modules['diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint'] = MagicMock()
sys.modules['diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_xl_inpaint'] = MagicMock()
sys.modules['xformers'] = MagicMock()
sys.modules['xformers.ops'] = MagicMock()
sys.modules['clip_interrogator'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['segment_anything'] = MagicMock()
sys.modules['insightface'] = MagicMock()
sys.modules['insightface.app'] = MagicMock()
sys.modules['gradio_client'] = MagicMock() # Often used by replicate/spaces
# Mocking torch can be risky if I need tensors, but let's assume I don't for now or use numpy.
# Soft features rely on numpy/cv2, not torch. 
# But Pipeline might import torch.
# If I leave torch real, it might try to load CUDA.
# Let's hope torch is available (CPU). It was satisfied in run.bat logs.

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.pipeline import Pipeline  # Assuming Pipeline is importable or I need to mock/extract methods

# Mocking Pipeline if it has heavy init
class MockPipeline:
    def _apply_mask_content(self, image, mask, mode):
        # Copy-paste logic or import if method is static/standalone?
        # Ideally import, but Pipeline might load models. 
        # Let's inspect pipeline.py again to see if I can import Pipeline without loading models.
        # It inherits from QThread usually?
        pass

# Actually, let's try to import the real Pipeline class but mock its init.
from unittest.mock import MagicMock, patch

class TestSoftLogic(unittest.TestCase):
    
    def setUp(self):
        # Patch init to avoid loading models
        with patch('core.pipeline.Pipeline.__init__', return_value=None):
            self.pipeline = Pipeline()

    def test_mask_content_fill(self):
        # Create a black image with a white hole
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        # Draw a white square in image
        cv2.rectangle(img, (40, 40), (60, 60), (255, 255, 255), -1)
        # Mask the square
        cv2.rectangle(mask, (40, 40), (60, 60), 255, -1)
        
        # 'fill' should inpaint the white square using surrounding black
        # Result should be close to black
        res = self.pipeline._apply_mask_content(img, mask, 'fill')
        
        # Center pixel (50,50) should be close to 0 (black), not 255
        center_val = res[50, 50]
        print(f"Fill Result Center: {center_val}")
        self.assertTrue(np.mean(center_val) < 50, "Fill should remove the white square")

    def test_mask_content_noise(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (60, 60), 255, -1)
        
        res = self.pipeline._apply_mask_content(img, mask, 'latent_noise')
        
        # Masked area should have noise (high variance)
        crop = res[40:60, 40:60]
        std_dev = np.std(crop)
        print(f"Noise StdDev: {std_dev}")
        self.assertTrue(std_dev > 10, "Latent Noise should have high variance")
        
        # Unmasked area should remain black
        self.assertEqual(res[10, 10, 0], 0)

    def test_pixel_composite_threshold(self):
        # Original: Gray
        orig = np.full((100, 100, 3), 100, dtype=np.uint8)
        # Inpainted: Gray + slight change
        inpainted = orig.copy()
        
        # Area A: Small change (below threshold) -> Should revert to Orig
        inpainted[10:30, 10:30] = 105 # Diff 5
        
        # Area B: Large change (above threshold) -> Should keep Inpainted
        inpainted[60:80, 60:80] = 200 # Diff 100
        
        mask = np.full((100, 100), 255, dtype=np.uint8)
        
        # Config for Soft Inpainting
        # Threshold 0.1 (approx 25 pixel value diff out of 255?)
        # Logic: float threshold 0.0-1.0. 
        # 0.1 * 255 = 25.5.
        # Diff 5 is < 25.5 -> Should revert.
        # Diff 100 is > 25.5 -> Should keep.
        
        dd_config = {
            'soft_diff_threshold': 0.1, # 0.1 * 255 = ~25
            'soft_mask_influence': 0.0,
            'soft_diff_contrast': 1.0
        }
        
        # Run composite
        # We need recovered 'orig_crop_img'. For test, passing 'orig' as both 'full_image' context 
        # is complex because method slices it. 
        # Method signature: _apply_pixel_composite(self, final_img, orig_full_img, M, new_w, new_h, dd_config)
        # This requires mimicking the affine transform.
        
        # Let's SIMPLIFY pipeline._apply_pixel_composite to rely on passed crops? 
        # No, the code does warpAffine internally.
        # I need to provide a dummy M (Identity)
        
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # Identity
        
        # Setup: final_img is the 'inpainted' result crop (100x100)
        # orig_full_img is the 'orig' image (100x100)
        # new_w, new_h = 100, 100
        
        res = self.pipeline._apply_pixel_composite(inpainted, orig, M, 100, 100, dd_config)
        
        # Check Area A (Small change) -> Should be close to Orig (100)
        val_a = res[20, 20]
        print(f"Composite Area A (Small Diff): {val_a}")
        self.assertTrue(np.allclose(val_a, [100, 100, 100], atol=2), "Should revert small changes")
        
        # Check Area B (Large change) -> Should be close to Inpainted (200)
        val_b = res[70, 70]
        print(f"Composite Area B (Large Diff): {val_b}")
        self.assertTrue(np.allclose(val_b, [200, 200, 200], atol=2), "Should keep large changes")

if __name__ == '__main__':
    unittest.main()
