import sys
import os
from unittest.mock import MagicMock, patch

# CRITICAL: Mock libs BEFORE anything else
mock_libs = [
    "torch", "torch.hub", "torch.nn", "torch.utils", "torch.utils.data", "torchvision",
    "diffusers", "transformers", "accelerate", "xformers", "xformers.ops",
    "ultralytics", "mediapipe", "insightface", "insightface.app",
    "basicsr", "nudenet", "requests", "tqdm", "compel", "segment_anything", "cv2"
]
for lib in mock_libs:
    if lib not in sys.modules:
        m = MagicMock()
        m.__path__ = []
        sys.modules[lib] = m

# Mock specific sub-attributes needed for imports
sys.modules['torch'].version = MagicMock()
sys.modules['torch'].__version__ = "2.1.0"
sys.modules['cv2'].INTER_LINEAR = 1
sys.modules['cv2'].COLOR_BGR2RGB = 4

import unittest
import numpy as np
from PIL import Image, PngImagePlugin
import piexif
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ComprehensiveVerification(unittest.TestCase):
    
    def test_01_ui_regression_presets_removed(self):
        """Verify that Style Presets buttons are removed from AdetailerUnitWidget"""
        print("\n[Step 1] Verifying UI (Style Presets removal)...")
        # Now import the UI code
        from ui.main_window_tabs import AdetailerUnitWidget
        from PyQt6.QtWidgets import QApplication
        
        # Ensure QApplication exists
        app = QApplication.instance() or QApplication(["test"])
        
        # Instantiate widget
        widget = AdetailerUnitWidget("Test Unit")
        
        # Check for preset buttons (should not exist as members)
        has_realism = hasattr(widget, 'btn_realism')
        has_anime = hasattr(widget, 'btn_anime')
        
        self.assertFalse(has_realism, "btn_realism should be deleted")
        self.assertFalse(has_anime, "btn_anime should be deleted")
        print("  [PASS] Style Presets removed from UI.")

    def test_02_metadata_preservation(self):
        """Verify robust EXIF/XMP preservation logic in save_image_with_metadata"""
        print("\n[Step 2] Verifying Metadata Preservation...")
        from core.metadata import save_image_with_metadata
        
        test_input = "comprehensive_input.png"
        test_output = "comprehensive_output.png"
        
        try:
            img = Image.new("RGB", (100, 100), color="blue")
            exif_dict = {"0th": {piexif.ImageIFD.Make: u"Verified Camera"}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            exif_bytes = piexif.dump(exif_dict)
            info = PngImagePlugin.PngInfo()
            info.add_text("XML:com.adobe.xmp", "<x:xmpmeta>Verified XMP</x:xmpmeta>")
            img.save(test_input, exif=exif_bytes, pnginfo=info)
            
            mock_config = MagicMock()
            mock_config.to_adetailer_json.return_value = {"test": "data"}
            
            cv2_img = np.zeros((100, 100, 3), dtype=np.uint8)
            save_image_with_metadata(cv2_img, test_input, test_output, mock_config)
            
            with Image.open(test_output) as out_img:
                raw_exif = out_img.info.get("exif")
                self.assertIsNotNone(raw_exif)
                out_exif = piexif.load(raw_exif)
                out_xmp = out_img.info.get("XML:com.adobe.xmp")
                
                self.assertEqual(out_exif["0th"][piexif.ImageIFD.Make], b"Verified Camera")
                self.assertIn("Verified XMP", out_xmp)
                self.assertEqual(out_exif["0th"][piexif.ImageIFD.Software], b"ObjectDetailer_Ultimate")
                
            print("  [PASS] Metadata preserved correctly.")
        finally:
            if os.path.exists(test_input): os.remove(test_input)
            if os.path.exists(test_output): os.remove(test_output)

    def test_03_pipeline_smoke(self):
        """Smoke test of the ImageProcessor pipeline flow"""
        print("\n[Step 3] Pipeline Smoke Test...")
        from core.pipeline import ImageProcessor
        
        # Already mocked ObjectDetector and ModelManager via global sys.modules
        processor = ImageProcessor(device="cpu")
        
        # Setup mocks on the instance
        processor.detector = MagicMock()
        processor.model_manager = MagicMock()
        
        processor.detector.detect.return_value = [{'box': [10, 10, 50, 50], 'conf': 0.9, 'class_id': 0}]
        mock_pipe_output = MagicMock()
        mock_pipe_output.images = [Image.new("RGB", (512, 512))]
        processor.model_manager.pipe.return_value = mock_pipe_output
        
        # Simple Config
        config = {
            'enabled': True, 'detector_model': 'face.pt', 'pos_prompt': 'face', 'neg_prompt': 'ugly',
            'dd_enabled': False, 'use_sam': False, 'use_soft_inpainting': False,
            'inpaint_width': 512, 'inpaint_height': 512, 'denoising_strength': 0.4,
            'crop_padding': 32, 'mask_blur': 4, 'mask_dilation': 4,
            'conf_thresh': 0.5, 'min_face_ratio': 0.0, 'max_face_ratio': 1.0, 'steps': 20, 'cfg_scale': 7.0,
            'controlnet_path': None
        }
        
        with patch.object(processor, '_get_lora_list', return_value=[]):
            result = processor.process(np.zeros((512, 512, 3), dtype=np.uint8), [config])
            self.assertIsNotNone(result)
            print("  [PASS] Pipeline smoke test successful.")

if __name__ == "__main__":
    unittest.main()
