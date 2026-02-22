import sys
import os
import unittest
import numpy as np
import cv2
import torch
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.color_fix import apply_color_fix
from core.pipeline import ImageProcessor
from core.config import AppConfig

class TestHiresFixColor(unittest.TestCase):
    def test_reforge_color_fix(self):
        """Test the new Reforge (LAB Histogram Matching) method."""
        # Create a source image (blue-ish)
        source = np.full((128, 128, 3), [255, 0, 0], dtype=np.uint8) # BGR
        # Create a target image (red-ish)
        target = np.full((128, 128, 3), [0, 0, 255], dtype=np.uint8) # BGR
        
        # Apply Reforge color fix
        result = apply_color_fix(target, source, method="Reforge")
        
        # Result should be closer to blue (source) than red (target)
        # We check the mean color
        mean_result = result.mean(axis=(0,1))
        # Blue channel should be higher than Red channel
        self.assertGreater(mean_result[0], mean_result[2])
        print(f"Reforge Color Fix Result Mean: {mean_result}")

    @patch('ui.main_window_tabs.cfg')
    @patch('ui.main_window_tabs.os.path.exists', return_value=True)
    def test_ui_get_config_captures_hires(self, mock_exists, mock_cfg):
        """Verify that UI correctly captures Hires Fix parameters."""
        from PyQt6.QtWidgets import QApplication, QWidget
        from ui.main_window_tabs import ADetailerUnitTab
        
        app = QApplication.instance() or QApplication([])
        
        # Mock config
        mock_cfg.get.return_value = 512
        mock_cfg.get_path.return_value = "fake_path"
        
        tab = ADetailerUnitTab("Unit 1", {})
        
        # Set values
        tab.chk_hires.setChecked(True)
        tab.combo_hires_upscaler.setCurrentText("Test_Upscaler.pth")
        
        # These use add_slider_row which we just fixed
        # Key: hires_upscale_factor
        for key, data in tab.settings.items():
            if key == 'hires_upscale_factor':
                data['widget'].setValue(2.0)
            elif key == 'hires_steps':
                data['widget'].setValue(15)
            elif key == 'hires_denoise':
                data['widget'].setValue(0.35)
        
        config = tab.get_config()
        
        self.assertTrue(config['use_hires_fix'])
        self.assertEqual(config['hires_upscaler'], "Test_Upscaler.pth")
        self.assertEqual(config['hires_upscale_factor'], 2.0)
        self.assertEqual(config['hires_steps'], 15)
        self.assertEqual(config['hires_denoise'], 0.35)
        print("UI get_config captured all Hires Fix parameters successfully.")

    @patch('core.pipeline.ModelManager')
    def test_pipeline_uses_esrgan_path(self, mock_model_manager):
        """Verify pipeline uses configurable ESRGAN path."""
        # Initialize ImageProcessor
        # We need to mock detector and upscaler initialization
        with patch('core.pipeline.ObjectDetector'), \
             patch('core.pipeline.ESRGANUpscaler'), \
             patch('core.pipeline.AppConfig') as MockConfig:
            
            MockConfig.return_value.get_path.side_effect = lambda cat, file=None: f"custom_{cat}_path"
            
            proc = ImageProcessor()
            proc.device = "cpu"
            
            # Setup inputs
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            box = [100, 100, 200, 200]
            kps = None
            padding = 1.0
            target_size = 512
            upscaler_name = "4x-UltraSharp.pth"
            
            # Call _apply_hires_upscale
            with patch('core.pipeline.os.path.join') as mock_join, \
                 patch('core.pipeline.align_and_crop', return_value=(image, None)):
                
                # Mock upscaler.load_model to return True
                proc.upscaler.load_model.return_value = True
                proc.upscaler.upscale.return_value = np.zeros((2048, 2048, 3), dtype=np.uint8)
                
                proc._apply_hires_upscale(image, box, kps, padding, target_size, upscaler_name, False)
                
                # Check if join was called with custom_esrgan_path
                # In _apply_hires_upscale: model_path = os.path.join(cfg.get_path('esrgan'), upscaler_name)
                # mock_join.assert_any_call("custom_esrgan_path", upscaler_name)
                # Let's check calls
                args_list = [call[0] for call in mock_join.call_args_list]
                self.assertTrue(any("custom_esrgan_path" in arg for arg in args_list))
                print("Pipeline used configurable ESRGAN path.")

if __name__ == '__main__':
    unittest.main()
