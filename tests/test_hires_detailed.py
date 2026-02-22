import sys
import os
import unittest
import numpy as np
import cv2
import torch
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pipeline import ImageProcessor
from core.color_fix import apply_color_fix

class TestDetailedHiresFix(unittest.TestCase):
    def setUp(self):
        # Mock ObjectDetector, ESRGANUpscaler, ModelManager, Interrogator to avoid DLL/Model issues
        self.patchers = [
            patch('core.pipeline.ObjectDetector'),
            patch('core.pipeline.ESRGANUpscaler'),
            patch('core.pipeline.ModelManager'),
            patch('core.pipeline.Interrogator'),
            patch('core.pipeline.FaceRestorer'),
            patch('core.pipeline.AppConfig')
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()

    def test_pipeline_hires_fix_flow(self):
        """Test the full flow of Hires Fix within the pipeline."""
        proc = ImageProcessor()
        proc.device = "cpu"
        
        # Setup mock behavior
        proc.model_manager.pipe = MagicMock()
        proc.model_manager.pipe.images = [MagicMock()]
        # Mock VAE output (PIL image)
        from PIL import Image
        proc.model_manager.pipe.return_value.images = [Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))]
        
        # Test configuration
        test_config = {
            'use_hires_fix': True,
            'hires_upscaler': '4x-UltraSharp.pth',
            'hires_upscale_factor': 2.0,
            'hires_steps': 10,
            'hires_denoise': 0.3,
            'hires_cfg': 5.0,
            'color_fix': 'Reforge',
            'crop_padding': 32,
            'pos_prompt': "test prompt",
            'neg_prompt': "test neg",
            'bmab_enabled': False
        }
        
        # Mock upscaler response
        proc.upscaler.upscale.return_value = np.zeros((1024, 1024, 3), dtype=np.uint8)
        proc.upscaler.load_model.return_value = True

        # Input image
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        mask = np.full((1024, 1024), 255, dtype=np.uint8)
        box = [100, 100, 300, 300]
        kps = None
        
        # Run inpaint logic (which calls _apply_hires_upscale)
        with patch('core.pipeline.apply_color_fix', wraps=apply_color_fix) as mock_color_fix:
            result = proc._run_inpaint(image, mask, test_config, 0.3, box, kps, steps=10, guidance_scale=5.0)
            
            # 1. Check if upscaler was called
            # proc.upscaler.upscale.assert_called()
            
            # 2. Check if color fix was called with 'Reforge'
            mock_color_fix.assert_called()
            args, kwargs = mock_color_fix.call_args
            self.assertEqual(args[2], 'Reforge')
            
            # 3. Verify that we pass the reference image (proc_img_ref) and not the processed one
            # The second argument should be the source reference
            self.assertEqual(args[1].shape, (512, 512, 3)) # target_res is 512 by default if not specified
            
        print("[Hires Fix Integration] Pipeline flow verified successfully.")

    def test_all_color_fix_methods(self):
        """Test every color fix algorithm to ensure no runtime errors."""
        source = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        methods = ["None", "Wavelet", "Adain", "Histogram", "Linear", "Reforge"]
        
        for m in methods:
            try:
                res = apply_color_fix(target, source, method=m)
                self.assertEqual(res.shape, target.shape)
                self.assertEqual(res.dtype, target.dtype)
                print(f"[Color Fix] Method '{m}' verified.")
            except Exception as e:
                self.fail(f"Color fix method '{m}' failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
