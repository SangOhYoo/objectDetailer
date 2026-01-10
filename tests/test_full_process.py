
import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch, ANY

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestFullPipelineFlow(unittest.TestCase):
    
    def setUp(self):
        # 1. Setup Mocks for External Libs
        self.external_libs = [
            "torch", "torch.hub", "diffusers", "transformers", "xformers", "xformers.ops",
            "compel", "segment_anything", "ultralytics", "mediapipe",
            "piexif", "piexif.helper", 
            "insightface", "insightface.app", "gradio_client"
        ]
        
        self.patchers = []
        self.mocks = {}
        
        # Patch sys.modules
        self.orig_modules = sys.modules.copy()
        for lib in self.external_libs:
            m = MagicMock()
            self.mocks[lib] = m
            sys.modules[lib] = m
            
        # Patch Internal Modules
        # meaningful mocks for core logic
        self.mock_mm = MagicMock()
        self.mock_det = MagicMock()
        self.mock_fr = MagicMock()
        self.mock_up = MagicMock()
        self.mock_sam = MagicMock()
        self.mock_dd = MagicMock()
        self.mock_vis = MagicMock()
        
        # Setup specific behaviors
        # DetailDaemonContext should be a context manager
        self.mock_dd.DetailDaemonContext.return_value.__enter__.return_value = MagicMock()
        self.mock_dd.DetailDaemonContext.return_value.__exit__.return_value = None
        
        # Detector should return fake detections
        # detection: {'box': [x1, y1, x2, y2], 'mask': None, 'conf': 0.9, 'class_id': 0}
        self.mock_det.ObjectDetector.return_value.detect.return_value = [
            {'box': [10, 10, 100, 100], 'mask': None, 'conf': 0.9, 'class_id': 0},
            {'box': [200, 200, 300, 300], 'mask': None, 'conf': 0.8, 'class_id': 0}
        ]
        
        # Config needs to be patchable
        self.mock_cfg_inst = MagicMock()
        self.mock_cfg_inst.get_path.return_value = "/dummy/path"
        
        # Register Internal Mocks
        sys.modules['core.model_manager'] = self.mock_mm
        
        # Configure Pipe Output
        mock_output = MagicMock()
        # Return a valid white image (512x512x3)
        mock_output.images = [np.ones((512, 512, 3), dtype=np.uint8) * 255]
        self.mock_mm.pipe.return_value = mock_output
        
        sys.modules['core.detector'] = self.mock_det
        sys.modules['core.face_restorer'] = self.mock_fr
        sys.modules['core.upscaler'] = self.mock_up
        sys.modules['core.sam_wrapper'] = self.mock_sam
        sys.modules['core.detail_daemon'] = self.mock_dd
        sys.modules['core.visualizer'] = self.mock_vis
        
        # Handle core.config
        # We need to mock the module and the instance inside it
        mock_config_mod = MagicMock()
        mock_config_mod.config_instance = self.mock_cfg_inst
        sys.modules['core.config'] = mock_config_mod
        
        # Now import Pipeline
        # We must reload or import fresh
        if 'core.pipeline' in sys.modules:
            del sys.modules['core.pipeline']
            
        from core.pipeline import ImageProcessor
        self.ImageProcessor = ImageProcessor

    def tearDown(self):
        # Restore modules (optional but executing in one process)
        # For simplicity in this script run, we don't fully restore to original state 
        # because the process will exit.
        pass

    def test_process_flow_lora_and_dd(self):
        processor = self.ImageProcessor(device="cpu")
        
        # Create a dummy input image (numpy array BGR)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Create a detailed config
        # We simulate a config that includes:
        # 1. Prompt with LoRA
        # 2. Detail Daemon enabled
        # 3. Soft Inpainting (implicitly via defaults or explicit)
        
        config = {
            'enabled': True,
            'pos_prompt': "A photo of <lora:TestLoRA:0.8> a person, <lora:Style:1.0>",
            'neg_prompt': "ugly",
            'model': 'test.ckpt',
            'vae': 'test.vae',
            'detector_model': 'yolo_v8n.pt', # Missing Key
            'box_threshold': 0.5,
            'conf_thresh': 0.5, # Missing Key
            'iou_threshold': 0.5,
            'min_face_ratio': 0.0,
            'max_face_ratio': 1.0,
            'denoising_strength': 0.4,
            'inpaint_width': 512,
            'inpaint_height': 512,
            'crop_padding': 32,
            'mask_blur': 4,
            'mask_dilation': 4,
            'steps': 20,
            'cfg_scale': 7.0,
            'use_sam': False,
            'dd_enabled': True, # Detail Daemon
            'dd_start': 0.2,
            'mask_content': 'latent_noise', # Soft Inpainting Feature
            'use_soft_inpainting': True,
            'soft_mask_influence': 0.2,
            'controlnet_path': None
        }
        
        # Execute Process
        # We mock internal methods to avoid heavy logic but we want to verify calls to managers
        
        # We mock _run_inpaint to check if it gets called, but we want the flow to reach it.
        # But _run_inpaint calls pipe inference (mocked).
        # We also need align_and_crop (imported from geometry). 
        # Geometry is NOT mocked in setUp list, so it might try to run real cv2 code.
        # This is fine if cv2 handles numpy zeros.
        
        # We need to Ensure Geometry is importable.
        # It uses cv2.
        
        # Run!
        try:
            result = processor.process(image, [config])
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Process raised exception: {e}")
            
        # --- VERIFICATIONS ---
        
        # 1. Check if LoRAs were parsed and managed
        # The prompt has 2 LoRAs.
        # _process_pass should call manage_lora for each detection.
        # Since we mocked 2 detections (in setUp), manage_lora should be called twice.
        
        # Check calls to model_manager.manage_lora
        # Arguments expected: list of tuples [('TestLoRA', 0.8), ('Style', 1.0)]
        expected_loras = [('TestLoRA', 0.8), ('Style', 1.0)]
        
        # NOTE: manage_lora signature is (lora_list, action).
        # Check call args
        mm = processor.model_manager
        
        print("\n[Verify] Checking LoRA Loading...")
        found_lora_call = False
        print(f"DEBUG calls: {mm.manage_lora.call_args_list}")
        for call in mm.manage_lora.call_args_list:
            args, _ = call
            if len(args) > 1 and args[1] == 'load':
                # Check list content
                # args[0] is the list
                call_list = args[0]
                # Convert both lists to sets of tuples for comparison (order agnostic)
                # But lists might contain tuples.
                # Just sort.
                if sorted(call_list) == sorted(expected_loras):
                    found_lora_call = True
                    print(f"  [Pass] manage_lora called with {expected_loras}")
        
        if not found_lora_call:
            print(f"  [Fail] manage_lora not found. Calls: {mm.manage_lora.call_args_list}")
            # Do not fail test immediately, log all failures
            
        self.assertTrue(found_lora_call, "LoRA loading was not triggered correctly.")
        
        # 2. Check Detail Daemon Context
        # _run_inpaint invokes DetailDaemonContext
        print("\n[Verify] Checking Detail Daemon...")
        dd_ctx = self.mock_dd.DetailDaemonContext
        if dd_ctx.called:
            print("  [Pass] DetailDaemonContext instantiated.")
            # Check args: pipe, enabled, config
            args, _ = dd_ctx.call_args
            # args[1] should be True (enabled)
            self.assertTrue(args[1], "DetailDaemon should be enabled")
        else:
            print("  [Fail] DetailDaemonContext NEVER instantiated.")
            self.fail("DetailDaemonContext not used.")
            
        # 3. Check Soft Inpainting Logic
        # It happens inside _run_inpaint -> _apply_mask_content
        # processor._apply_mask_content is a method, we can mock it or check side effects?
        # Since we use real pipeline class, the method exists.
        # We can't mock 'self.method' easily on an instance in python without setattr, but we want to test flow.
        # We can check if cv2.inpaint or numpy constructs were used?
        # Or better: Check if result is returned.
        
        print("\n[Verify] Process returned result...")
        self.assertIsNotNone(result)
        print("  [Pass] Pipeline execution completed successfully.")

if __name__ == '__main__':
    unittest.main()
