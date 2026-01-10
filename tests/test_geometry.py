
import unittest
import math
import numpy as np
from core.geometry import get_rotation_angle, rotate_point, is_anatomically_correct

class TestGeometry(unittest.TestCase):
    
    def test_rotate_point(self):
        # Rotate point (10, 0) around (0, 0) by 90 degrees -> should be roughly (0, 10)
        # Note: image coordinates, y is down.
        # x' = x cos - y sin, y' = x sin + y cos
        # 10 * 0 - 0 * 1 = 0
        # 10 * 1 + 0 * 0 = 10
        pt = [10, 0]
        center = [0, 0]
        angle = 90
        rotated = rotate_point(pt, angle, center)
        np.testing.assert_allclose(rotated, [0, 10], atol=1e-5)

    def test_get_rotation_angle_horizontal(self):
        # Eyes horizontally aligned
        kps = [
            [10, 10], # LE
            [30, 10], # RE
            [20, 20], # Nose
            [10, 30], # LM
            [30, 30]  # RM
        ]
        angle = get_rotation_angle(kps)
        self.assertAlmostEqual(angle, 0.0)

    def test_get_rotation_angle_vertical(self):
        # Eyes vertically aligned (90 deg rotation)
        # Left Eye (originally left) is now at bottom (10, 30)
        # Right Eye (originally right) is now at top (10, 10)
        # RE - LE vector: (0, -20) -> atan2(-20, 0) = -90 degrees
        
        # Wait, let's trace coordinate system
        # Standard: LE(10,10) RE(30,10). dx=20, dy=0. atan2(0, 20) = 0.
        
        # Rotated 90 deg clockwise? 
        # LE moves to (10, 10), RE moves to (10, 30). dx=0, dy=20. atan2(20,0) = 90.
        kps = [
            [10, 10], # LE 
            [10, 30], # RE
            [0, 20],  # Nose (to the left relative to eyes? No, nose should be 'below' eyes)
            # If eyes are vertical x=10, nose "below" means x < 10 or x > 10?
            # Let's assume standard face, nose is y+ relative to eyes.
            # Rotated 90 deg CW, nose is x- relative to eyes.
            # So Nose at (0, 20) seems right for 90 deg.
             
            [0, 10], # LM
            [0, 30]  # RM
        ]
        # Cross product check:
        # ex = 0, ey = 20
        # nx = 0-10 = -10, ny = 20-10 = 10
        # CP = 0*10 - 20*(-10) = 200 > 0.
        
        angle = get_rotation_angle(kps)
        self.assertAlmostEqual(angle, 90.0)

    def test_is_anatomically_correct(self):
        # Standard face
        kps_correct = [
            [10, 10], [30, 10], # Eyes
            [20, 20],           # Nose
            [10, 30], [30, 30]  # Mouth
        ]
        self.assertTrue(is_anatomically_correct(kps_correct))
        
        # Monster face (Mouth above Nose)
        kps_wrong = [
            [10, 10], [30, 10], # Eyes
            [20, 30],           # Nose (Low)
            [10, 20], [30, 20]  # Mouth (High)
        ]
        self.assertFalse(is_anatomically_correct(kps_wrong))
        
        # Monster face (Nose above Eyes)
        kps_nose_high = [
            [10, 20], [30, 20], # Eyes (Low)
            [20, 10],           # Nose (High)
            [10, 30], [30, 30]  # Mouth
        ]
        self.assertFalse(is_anatomically_correct(kps_nose_high))

if __name__ == '__main__':
    unittest.main()
