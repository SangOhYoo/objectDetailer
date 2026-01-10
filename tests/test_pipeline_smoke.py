
import unittest
import sys
from unittest.mock import MagicMock, patch

class TestPipelineSmoke(unittest.TestCase):
    
    def setUp(self):
        # 1. Setup Mocks for External Libs
        self.external_libs = [
            "torch", "torch.hub", "diffusers", "transformers", "xformers", "xformers.ops",
            "compel", "segment_anything", "ultralytics", "mediapipe",
            "piexif", "piexif.helper"
        ]
        
        # 2. Setup Mocks for Internal Core Modules (to isolate pipeline)
        self.internal_modules = [
            "core.model_manager",
            "core.detector",
            "core.face_restorer",
            "core.upscaler",
            "core.sam_wrapper",
            "core.detail_daemon",
            # We don't mock simple utils if possible, but Visualizer might use cv2/numpy
            "core.visualizer", 
        ]
        
        self.patchers = []
        
        # Create a consistent Mock for config since it is imported as 'from core.config import config_instance'
        self.mock_cfg_instance = MagicMock()
        self.mock_cfg_instance.get_path.return_value = "dummy"
        self.mock_config_mod = MagicMock()
        self.mock_config_mod.config_instance = self.mock_cfg_instance
        
        # Patch system modules
        custom_mocks = {
            "core.config": self.mock_config_mod
        }
        
        for lib in self.external_libs:
            custom_mocks[lib] = MagicMock()
            
        for mod in self.internal_modules:
            custom_mocks[mod] = MagicMock()
            
        self.sys_modules_patcher = patch.dict(sys.modules, custom_mocks)
        self.sys_modules_patcher.start()
        
        # Force reload of core.pipeline to pick up mocks
        if 'core.pipeline' in sys.modules:
            del sys.modules['core.pipeline']

    def tearDown(self):
        self.sys_modules_patcher.stop()

    def test_pipeline_init(self):
        # Import under test
        try:
            from core.pipeline import ImageProcessor
            # Also need to ensure the classes used are the Mocks we injected
            from core.model_manager import ModelManager
        except ImportError as e:
            self.fail(f"Import failed: {e}")
        
        # Instantiate
        processor = ImageProcessor(device="cpu")
        
        # Verification
        # ModelManager should be the Mock we injected into sys.modules
        # We can access it via the module import above (since it comes from sys.modules)
        ModelManager.assert_called()
        self.assertIsNotNone(processor)
        
        # Verify it has expected attributes
        self.assertTrue(hasattr(processor, 'model_manager'))

if __name__ == '__main__':
    unittest.main()
