# Mock diffusers before ANY package imports that use it
import sys
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import torch

# 1. Mock diffusers entirely to bypass DLL load errors in test env
mock_diffusers = MagicMock()
sys.modules["diffusers"] = mock_diffusers

# Add project root
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import ImageProcessor
from core.config import config_instance as cfg
from PIL import Image

def run_comprehensive_test():
    print("=== Starting Comprehensive System Test ===")
    
    # 1. Setup Mock environment or actual device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    
    # Create a dummy image for testing
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (412, 412), (200, 200, 200), -1) # Simple box
    
    # 2. Configure a multi-feature pass
    test_config = {
        'enabled': True,
        'unit_name': 'TestUnit',
        'detector_model': 'face_yolov8n.pt',
        'conf_thresh': 0.3,
        'denoising_strength': 0.5,
        'pos_prompt': 'detailed face, high quality',
        'neg_prompt': 'blurry, distorted',
        'use_sam': False,
        'auto_rotate': True,
        'crop_padding': 32,
        'inpaint_width': 512,
        'inpaint_height': 512,
        # New Features to test
        'use_pose_guide': True,
        'color_fix': 'Histogram',
        'post_sharpen': 0.8,
        'restore_face': True,
        'restore_face_strength': 0.7,
        'steps': 1, # Minimal steps for speed
        'cfg_scale': 7.0,
        'sampler_name': 'Euler a',
        'sep_ckpt': False,
        'use_hires_fix': False,
        # Missing essential keys
        'min_face_ratio': 0.0,
        'max_face_ratio': 1.0,
        'bmab_landscape_detail': False,
        'mask_merge_mode': 'None',
        'inpaint_full_res': False,
        'dd_enabled': False,
        'bmab_enabled': False,
        'bmab_edge_enabled': False,
        'resize_enable': False,
        'context_expand_factor': 1.0,
        'mask_content': 'original',
        'guidance_start': 0.0,
        'guidance_end': 1.0,
        'control_weight': 1.0,
        'control_model': 'None',
        'mask_dilation': 4,
        'mask_blur': 4,
        'mask_erosion': 0,
        'auto_rotate_padding': 0,
        'noise_multiplier': 1.0,
        'context_expand_factor': 1.0,
        'inpaint_area': 'Only Masked',
        'use_controlnet': True,
        'auto_prompting': True,
        'interrogator_threshold': 0.35
    }

    # 3. Patching heavy components to avoid OOM or missing models during automated CI
    # We want to verify logic flow, not necessarily AI output quality here.
    with patch('core.model_manager.ModelManager.load_sd_model'), \
         patch('core.model_manager.ModelManager.apply_scheduler'), \
         patch('core.model_manager.ModelManager.manage_lora'), \
         patch('core.detector.ObjectDetector.detect') as mock_detect, \
         patch('core.detector.ObjectDetector.detect_pose') as mock_pose_detect, \
         patch('core.face_restorer.FaceRestorer.restore') as mock_restore, \
         patch('core.interrogator.Interrogator.interrogate') as mock_interrogate:
        
        # Setup Mocks
        # Simulate 1 face detection (with all expected keys)
        mock_detect.return_value = [{
            'box': [100, 100, 200, 200], 
            'conf': 0.9, 
            'label': 0, 
            'label_name': 'face',
            'mask': None,
            'kps': None
        }]
        # Simulate 1 pose detection
        mock_pose_detect.return_value = [{'box': [0, 0, 512, 512], 'keypoints': np.random.rand(17, 3) * 512}]
        # Simulate face restoration (identity return)
        mock_restore.side_effect = lambda img, strength: img
        # Simulate interrogation
        mock_interrogate.return_value = "1girl, solo, smile"

        # Mock pipeling inference call
        mock_pipe = MagicMock()
        # Ensure hasattr(mock_pipe, "controlnet") is True for pipeline logic
        mock_pipe.controlnet = MagicMock() 
        mock_pipe.components = {}
        mock_pipe.vae.enable_tiling = MagicMock()
        mock_pipe.vae.enable_slicing = MagicMock()
        
        # Mocking the actual __call__ of the pipeline
        dummy_output = MagicMock()
        dummy_output.images = [Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))]
        mock_pipe.return_value = dummy_output

        print("Patching ModelManager...")
        processor = ImageProcessor(device)
        processor.model_manager.pipe = mock_pipe
        
        # 4. Run Process
        print("Executing Pipeline Process Loop...")
        result = processor.process(img, [test_config])
        
        # 5. Verifiction
        print("\n--- Verification Report ---")
        
        # Check if detect_pose was called (Pose Guide logic check)
        if mock_pose_detect.called:
            print("[Check] Pose Detection: PASSED (Triggered by use_pose_guide)")
        else:
            print("[Check] Pose Detection: FAILED (Not triggered)")
            
        # Check if FaceRestorer was called
        if mock_restore.called:
            print("[Check] Face Restoration: PASSED")
        else:
            print("[Check] Face Restoration: FAILED")
            
        # Check if Interrogator was called
        if mock_interrogate.called:
            print(f"[Check] Interrogator: PASSED (Tags: {mock_interrogate.return_value})")
        else:
            print("[Check] Interrogator: FAILED")

        print("=== System Test Complete ===")

from PIL import Image
if __name__ == "__main__":
    run_comprehensive_test()
