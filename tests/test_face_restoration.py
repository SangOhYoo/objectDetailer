import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock
from core.face_restorer import FaceRestorer

# Mock config to avoid loading real config file
import core.config
core.config.config_instance = MagicMock()
core.config.config_instance.get_path.return_value = "dummy"

class TestFaceRestoration:
    def test_restore_blending(self):
        """
        Test that Blend Strength correctly mixes Original and Restored images.
        """
        restorer = FaceRestorer(device='cpu')
        
        # Mock GFPGANer
        mock_gfpgan = MagicMock()
        restorer.gfpgan = mock_gfpgan
        # Mock load_model to return True
        restorer.load_model = MagicMock(return_value=True)
        
        # Input: Black Image (0)
        input_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Restored Output: White Image (255)
        restored_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Configure Mock to return our restored image
        # enhance returns (cropped_faces, restored_faces, restored_img)
        mock_gfpgan.enhance.return_value = (None, None, restored_img)
        
        # Case 1: Strength 1.0 (Full Restore) -> Expect White (255)
        res_full = restorer.restore(input_img, strength=1.0)
        assert np.mean(res_full) == 255
        
        # Case 2: Strength 0.0 (Original) -> Expect Black (0)
        res_zero = restorer.restore(input_img, strength=0.0)
        assert np.mean(res_zero) == 0
        
        # Case 3: Strength 0.5 (Blend) -> Expect Gray (127 or 128)
        res_half = restorer.restore(input_img, strength=0.5)
        mean_val = np.mean(res_half)
        print(f"Mean value for 0.5 strength: {mean_val}")
        
        # Allow small rounding error
        assert 126 <= mean_val <= 129

if __name__ == "__main__":
    pytest.main([__file__])
