import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestUIConfig(unittest.TestCase):
    @patch('ui.main_window_tabs.cfg')
    @patch('ui.main_window_tabs.os.path.exists', return_value=True)
    def test_ui_get_config_captures_hires(self, mock_exists, mock_cfg):
        """Verify that UI correctly captures Hires Fix parameters."""
        # Mock PyQt6 to avoid needing a display
        from PyQt6.QtWidgets import QApplication, QWidget
        from ui.main_window_tabs import ADetailerUnitTab
        
        # Use QOffscreenSurface or just a dummy app if possible
        app = QApplication.instance() or QApplication(["test"])
        
        # Mock config
        mock_cfg.get.return_value = 512
        mock_cfg.get_path.return_value = "fake_path"
        
        tab = ADetailerUnitTab("Unit 1", {})
        
        # Set values
        tab.chk_hires.setChecked(True)
        tab.combo_hires_upscaler.setCurrentText("Test_Upscaler.pth")
        tab.combo_color_fix.setCurrentText("Reforge")
        
        # Check if sliders are working (manual trigger of value change if needed)
        # Key: hires_upscale_factor (index depends on creation order, but key is in self.settings)
        self.assertIn('hires_upscale_factor', tab.settings)
        self.assertIn('hires_steps', tab.settings)
        self.assertIn('hires_denoise', tab.settings)
        self.assertIn('hires_cfg', tab.settings)
        
        tab.settings['hires_upscale_factor']['widget'].setValue(2.0)
        tab.settings['hires_steps']['widget'].setValue(15)
        tab.settings['hires_denoise']['widget'].setValue(0.35)
        tab.settings['hires_cfg']['widget'].setValue(5.5)
        
        config = tab.get_config()
        
        self.assertTrue(config['use_hires_fix'])
        self.assertEqual(config['hires_upscaler'], "Test_Upscaler.pth")
        self.assertEqual(config['color_fix'], "Reforge")
        self.assertEqual(config['hires_upscale_factor'], 2.0)
        self.assertEqual(config['hires_steps'], 15)
        self.assertEqual(config['hires_denoise'], 0.35)
        self.assertEqual(config['hires_cfg'], 5.5)
        
        print("UI get_config captured all Hires Fix parameters successfully: PASSED")

if __name__ == '__main__':
    unittest.main()
