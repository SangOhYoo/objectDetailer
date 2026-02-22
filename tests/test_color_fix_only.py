import sys
import os
import unittest
import numpy as np
import cv2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.color_fix import apply_color_fix

class TestColorFixOnly(unittest.TestCase):
    def test_reforge_color_fix(self):
        """Test the new Reforge (LAB Histogram Matching) method in isolation."""
        # Create a source image (blue-ish)
        source = np.full((128, 128, 3), [255, 0, 0], dtype=np.uint8) # BGR
        # Create a target image (red-ish)
        target = np.full((128, 128, 3), [0, 0, 255], dtype=np.uint8) # BGR
        
        # Apply Reforge color fix
        result = apply_color_fix(target, source, method="Reforge")
        
        # Result should be blue (source)
        mean_result = result.mean(axis=(0,1))
        # Blue channel should be higher than Red channel
        self.assertGreater(mean_result[0], mean_result[2])
        print(f"Reforge Color Fix Result Mean (Isolation): {mean_result}")
        print("Color Fix Logic: PASSED")

if __name__ == '__main__':
    unittest.main()
