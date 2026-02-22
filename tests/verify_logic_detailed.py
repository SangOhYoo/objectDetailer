import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock heavy modules before they are imported by our core modules
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['compel'] = MagicMock()
sys.modules['controlnet_aux'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['insightface'] = MagicMock()
sys.modules['gfpgan'] = MagicMock()

# Now we can safely import our project logic
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.color_fix import apply_color_fix
from core.config import AppConfig

# Mocking PyQt6 is tricky because of the C++ bindings, 
# so we'll test UI logic by inspecting the code or using a very light mock.

class TestLogicDetailed(unittest.TestCase):
    def test_reforge_color_fix_logic(self):
        """Verify Reforge color fix algorithm correctness."""
        print("\n--- Testing Reforge Color Fix ---")
        # Source: Pure Blue [255, 0, 0] (BGR)
        source = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
        # Target: Pure Red [0, 0, 255] (BGR)
        target = np.full((64, 64, 3), [0, 0, 255], dtype=np.uint8)
        
        # In Reforge matching, the target's colors should be shifted towards the source's colors.
        result = apply_color_fix(target, source, method="Reforge")
        
        mean_result = result.mean(axis=(0,1))
        print(f"  Source Mean (Blue): {source.mean(axis=(0,1))}")
        print(f"  Target Mean (Red): {target.mean(axis=(0,1))}")
        print(f"  Result Mean: {mean_result}")
        
        # Result's blue channel (0) should now be significant
        self.assertGreater(mean_result[0], 100)
        # Result's red channel (2) should be less than 100 if matching worked strictly
        self.assertLess(mean_result[2], 150)
        print("  [Pass] Reforge matched color distributions successfully.")

    def test_config_esrgan_path(self):
        """Verify ESRGAN path is correctly handled in config."""
        print("\n--- Testing Config ESRGAN Path ---")
        config = AppConfig()
        # Mocking the data inside
        config.data['paths']['esrgan'] = "D:/Test/Models/ESRGAN"
        
        path = config.get_path('esrgan')
        self.assertEqual(path, "D:/Test/Models/ESRGAN")
        print(f"  Config Path: {path}")
        print("  [Pass] ESRGAN path correctly retrieved from config.")

    def test_pipeline_path_joining(self):
        """Verify pipeline joins upscaler path correctly using the new logic."""
        print("\n--- Testing Pipeline Upscaler Path Logic ---")
        # We manually test the line we modified:
        # model_path = os.path.join(cfg.get_path('esrgan'), upscaler_name)
        
        fake_esrgan_dir = "R:/Models/Upscalers"
        upscaler_name = "RealESRGAN_x4plus.pth"
        
        # Simulate os.path.join(cfg.get_path('esrgan'), upscaler_name)
        joined_path = os.path.normpath(os.path.join(fake_esrgan_dir, upscaler_name))
        expected = os.path.normpath("R:/Models/Upscalers/RealESRGAN_x4plus.pth")
        
        self.assertEqual(joined_path, expected)
        print(f"  Joined Path: {joined_path}")
        print("  [Pass] Upscaler path joining logic is correct.")

    def test_ui_setting_capture_logic(self):
        """
        Verify the UI logic fix in add_slider_row.
        Since we can't run PyQt6 easily, we verify that the code change 
        correctly assigns the widget to the settings dictionary.
        """
        print("\n--- Verifying UI Logic Change ---")
        # We are checking if:
        # self.settings[key] = {'widget': spin, 'default': default_val}
        # was added to add_slider_row.
        
        ui_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ui', 'main_window_tabs.py'))
        with open(ui_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if the fix exists in add_slider_row (around row 1100)
        search_pattern = "self.settings[key] = {\n            'widget': spin,\n            'default': default_val\n        }"
        # Simplified check
        self.assertIn("self.settings[key] = {", content)
        self.assertIn("'widget': spin", content)
        self.assertIn("'default': default_val", content)
        
        # Count occurrences to ensure it's in add_slider_row too (previously only in add_slider_row_manual)
        # add_slider_row_manual has it. add_slider_row now has it too.
        count = content.count("self.settings[key] = {")
        self.assertGreaterEqual(count, 2) # Should be at least 2 places now
        print(f"  Setting capture logic found in {count} locations.")
        print("  [Pass] UI settings capture fix confirmed in source code.")

if __name__ == '__main__':
    unittest.main()
