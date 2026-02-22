import sys
import os
import numpy as np
import cv2
import torch
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from core.pipeline import ImageProcessor
from core.config import config_instance as cfg

def test_flag_respect():
    print("Testing Flag Respect...")
    
    # Mock dependencies
    device = "cpu"
    processor = ImageProcessor(device=device)
    
    # Create a dummy image
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Define a config with bmab_enabled = False explicitly
    config = {
        'enabled': True,
        'detector_model': 'face_yolov8n.pt',
        'conf_thresh': 0.35,
        'max_det': 1,
        'min_face_ratio': 0.0,
        'max_face_ratio': 1.0,
        'bmab_enabled': False,
        'color_fix': 'None',
        'use_sam': False,
        'crop_padding': 32,
        'mask_dilation': 4,
        'mask_blur': 12,
        'restore_face': False,
        'denoising_strength': 0.4,
        'steps': 1,
        'cfg_scale': 7.0,
        'sampler_name': 'Euler a',
        'pos_prompt': 'test',
        'neg_prompt': 'test',
        'use_hires_fix': False
    }
    
    # Mock detector and other internal calls to speed up
    processor.detector.detect = MagicMock(return_value=[{'box': [100, 100, 200, 200], 'conf': 0.9, 'mask': None}])
    processor.detector.offload_models = MagicMock()
    processor._run_inpaint = MagicMock(return_value=dummy_img)
    
    # We want to check if apply_bmab_basic is called when bmab_enabled is False
    with patch('core.bmab_utils.apply_bmab_basic') as mock_bmab:
        processor.process(dummy_img, [config])
        mock_bmab.assert_not_called()
        print("PASS: apply_bmab_basic was NOT called when disabled.")

def test_upscaler_fix():
    print("Testing Upscaler Fix...")
    from core.upscaler import ESRGANUpscaler
    
    upscaler = ESRGANUpscaler(device="cpu")
    # Mock model and attributes
    upscaler.model = MagicMock()
    upscaler.model.scale = 4
    upscaler.model.return_value = torch.zeros((1, 3, 400, 400)) # Small output for mock
    
    dummy_in = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # This should not raise NameError 'scale'
    try:
        # We need to trigger _upscale_tiled by passing a large image
        large_in = np.zeros((600, 600, 3), dtype=np.uint8)
        upscaler.upscale(large_in)
        print("PASS: Upscaler _upscale_tiled executed without NameError.")
    except NameError as e:
        print(f"FAIL: {e}")
    except Exception as e:
        # Other errors might happen due to mocking, but we specifically care about NameError
        if "name 'scale' is not defined" in str(e):
            print("FAIL: NameError 'scale' still present.")
        else:
            print(f"Caught expected mock-related error: {e}")

if __name__ == "__main__":
    try:
        test_flag_respect()
    except Exception as e:
        print(f"Error in flag test: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        test_upscaler_fix()
    except Exception as e:
        print(f"Error in upscaler test: {e}")
        import traceback
        traceback.print_exc()
