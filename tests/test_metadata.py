
import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
from core.metadata import save_image_with_metadata

class TestMetadata(unittest.TestCase):
    
    @patch('core.metadata.Image')
    @patch('core.metadata.piexif')
    @patch('os.path.exists')
    def test_save_image_with_metadata(self, mock_exists, mock_piexif, mock_image):
        # Setup mocks
        mock_exists.return_value = False # Simulate no original file
        
        mock_cv2_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_config = MagicMock()
        mock_config.to_adetailer_json.return_value = {
            "pos_prompt": "beautiful face",
            "neg_prompt": "ugly",
            "steps": 20
        }
        
        mock_pil_img = MagicMock()
        mock_image.fromarray.return_value = mock_pil_img
        
        # Test execution
        result = save_image_with_metadata(
            cv2_image=mock_cv2_img,
            original_path="dummy_orig.png",
            save_path="dummy_out.png",
            config=mock_config
        )
        
        # Verify
        self.assertTrue(result)
        
        # Verify save was called
        mock_pil_img.save.assert_called_once()
        
        # Verify PngInfo was populated
        call_args = mock_pil_img.save.call_args
        self.assertIsNotNone(call_args)
        kwargs = call_args[1]
        self.assertIn('pnginfo', kwargs)
        
        pnginfo = kwargs['pnginfo']
        # We can't easily inspect PngInfo object in mock without side methods, 
        # but we know it should be passed.
        
        # Verify EXIF construction attempted
        mock_piexif.dump.assert_called()

    @patch('core.metadata.Image')
    def test_save_fail_resorts_to_cv2(self, mock_image):
        # Simulate PIL Error
        mock_image.fromarray.side_effect = Exception("PIL Error")
        
        mock_cv2_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch('core.metadata.imwrite') as mock_imwrite:
            result = save_image_with_metadata(
                cv2_image=mock_cv2_img,
                original_path="dummy.png",
                save_path="output.png",
                config=MagicMock()
            )
            
            # Should return False (since metadata save failed) 
            # OR logic in function: returns False if exception caught.
            self.assertFalse(result)
            
            # But should attempt cv2 save
            mock_imwrite.assert_called_once()

if __name__ == '__main__':
    unittest.main()
