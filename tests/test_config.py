
import unittest
import os
import yaml
import tempfile
from core.config import AppConfig

class TestAppConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.test_dir.name, "test_config.yaml")
        
        # Reset singleton instance for each test to ensure fresh load
        AppConfig._instance = None
        
    def tearDown(self):
        self.test_dir.cleanup()
        AppConfig._instance = None

    def test_default_config_generation(self):
        # Initialize config where file doesn't exist
        config = AppConfig(self.config_path)
        
        # Check if file was created
        self.assertTrue(os.path.exists(self.config_path))
        
        # Check default values
        self.assertEqual(config.get("system", "log_level"), "DEBUG")
        self.assertEqual(config.get("defaults", "resolution"), 512)

    def test_load_existing_config(self):
        # Create a dummy config
        data = {
            "system": {
                "log_level": "INFO"
            },
            "defaults": {
                "resolution": 1024
            }
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
            
        config = AppConfig(self.config_path)
        
        # Verify loaded values
        self.assertEqual(config.get("system", "log_level"), "INFO")
        self.assertEqual(config.get("defaults", "resolution"), 1024)
        
        # Verify defaults are merged (missing keys should be present)
        # padding is in defaults but not in our dummy file
        self.assertEqual(config.get("defaults", "padding"), 1.5)

    def test_save_config(self):
        config = AppConfig(self.config_path)
        
        # Modify and save
        success = config.save_config({"new_section": {"key": "value"}})
        self.assertTrue(success)
        
        # Verify file content
        with open(self.config_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertEqual(saved_data["new_section"]["key"], "value")

if __name__ == '__main__':
    unittest.main()
