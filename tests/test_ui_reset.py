
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from PyQt6.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Config to avoid loading real config.yaml which might depend on local paths
with patch('core.config.config_instance') as mock_cfg:
    mock_cfg.get.return_value = {}
    mock_cfg.get_path.return_value = ""
    # Ensure ui.main_window_tabs can be imported even if it uses config
    from ui.main_window_tabs import AdetailerUnitWidget

app = QApplication(sys.argv)

class TestUIReset(unittest.TestCase):
    def setUp(self):
        self.widget = AdetailerUnitWidget("Test Unit")
    
    @patch('PyQt6.QtWidgets.QMessageBox.question')
    def test_reset_functionality(self, mock_question):
        # 1. Setup proper mock for MessageBox to return "Yes"
        from PyQt6.QtWidgets import QMessageBox
        mock_question.return_value = QMessageBox.StandardButton.Yes
        
        # 2. Modify some values
        # Slider/Spinbox (Confidence)
        self.widget.settings['conf_thresh']['widget'].setValue(0.99)
        self.assertNotEqual(self.widget.settings['conf_thresh']['widget'].value(), 0.35)
        
        # BMAB Slider
        self.widget.settings['bmab_contrast']['widget'].setValue(2.0)
        
        # Checkbox (Enable)
        self.widget.chk_enable.setChecked(False) # Default for "Test Unit" might be False? 
        # "1" in unit_name checks default. "Test Unit" -> default False.
        # Let's set it to True to test reset (which should set it to False for non-Pass1)
        self.widget.chk_enable.setChecked(True)
        
        # 3. Call Reset
        self.widget.on_reset_clicked()
        
        # 4. Verify Defaults
        # Confidence should be 0.35 (default in code)
        self.assertAlmostEqual(self.widget.settings['conf_thresh']['widget'].value(), 0.35)
        
        # BMAB Contrast should be 1.0
        self.assertAlmostEqual(self.widget.settings['bmab_contrast']['widget'].value(), 1.0)
        
        # Enable should be False (since "Test Unit" != "Pass 1")
        self.assertFalse(self.widget.chk_enable.isChecked())
        
        print("UI Reset Logic Verified Successfully")

if __name__ == "__main__":
    unittest.main()
