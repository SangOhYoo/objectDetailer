
import os
import sys
import torch
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_manager import ModelManager
from core.config import config_instance as cfg

class TestLoraManager(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.mm = ModelManager(self.device)
        self.mm.pipe = MagicMock()
        # Mocking config to return a dummy lora path
        cfg.get_path = MagicMock(return_value="D:/AI_Models/Lora")

    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test_manage_lora_unload_on_empty(self, mock_isfile, mock_exists):
        """Verify that manage_lora unloads when lora_list is empty."""
        mock_exists.return_value = True
        mock_isfile.return_value = True
        
        # 1. Load some LoRA
        lora_list = [("test_lora", 0.7)]
        with patch.object(self.mm, '_load_new_loras') as mock_load:
            self.mm.manage_lora(lora_list, "load")
            self.assertEqual(self.mm.loaded_loras, lora_list)
        
        # 2. Call with empty list - should UNLOAD
        with patch.object(self.mm, '_unload_all') as mock_unload:
            self.mm.manage_lora([], "load")
            mock_unload.assert_called_once()
            self.assertEqual(self.mm.loaded_loras, [])

    @patch('os.walk')
    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test_recursive_search_containment(self, mock_isfile, mock_exists, mock_walk):
        """Verify that recursive search uses containment check."""
        mock_exists.side_effect = lambda p: "test_lora" in p
        mock_isfile.side_effect = lambda p: "test_lora" in p
        
        # Mock os.walk to find a LoRA with a prefix
        mock_walk.return_value = [
            ("D:/AI_Models/Lora/subdir", [], ["prefix_test_lora.safetensors"])
        ]
        
        lora_list = [("test_lora", 0.7)]
        
        with patch.object(self.mm.pipe, 'load_lora_weights') as mock_load_weights:
            self.mm._load_new_loras(lora_list, "D:/AI_Models/Lora")
            
            # Check if it found the file with the prefix
            expected_path = os.path.join("D:/AI_Models/Lora/subdir", "prefix_test_lora.safetensors")
            # Normalize for comparison if needed, but here we just check if it was called
            args, kwargs = mock_load_weights.call_args
            self.assertIn("prefix_test_lora.safetensors", args[0])

if __name__ == "__main__":
    unittest.main()
