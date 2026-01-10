
import sys
import os
from PyQt6.QtWidgets import QApplication

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_controlnet_wiring():
    app = QApplication(sys.argv)
    
    try:
        from ui.main_window_tabs import AdetailerUnitWidget
        
        print("[Test] Creating AdetailerUnitWidget...")
        widget = AdetailerUnitWidget(unit_name="TestUnit")
        
        # Simulate User Interaction
        print("[Test] Setting ControlNet Params...")
        
        # 1. Slider: Weight
        # add_slider_row stores widget in self.settings['control_weight']['widget']
        # But we can also access via direct variable? No, add_slider_row doesn't create self.var usually.
        # It puts it in self.settings.
        
        if 'control_weight' in widget.settings:
            # Note: The widget stored in settings is the QDoubleSpinBox (or QSpinBox)
            # NOT the QSlider. QSlider updates QSpinBox via signals.
            spin = widget.settings['control_weight']['widget']
            spin.setValue(0.75) 
            print(f"     Set Weight SpinBox to 0.75")
        else:
            print("[Fail] 'control_weight' widget not found in settings!")
            
        # 2. ComboBox: Model
        # This is strictly named 'self.combo_cn_model'
        widget.combo_cn_model.addItem("Test_Model_v1.pth")
        widget.combo_cn_model.setCurrentText("Test_Model_v1.pth")
        print(f"     Set Model to 'Test_Model_v1.pth'")
        
        # 3. ComboBox: Module (Preprocessor)
        widget.combo_cn_module.setCurrentText("canny")
        print(f"     Set Preprocessor to 'canny'")
        
        # Retrieve Config
        cfg = widget.get_config()
        
        print("\n[Test] Retrieved Config:")
        print(f"     Control Model: {cfg['control_model']}")
        print(f"     Control Module: {cfg['control_module']}")
        print(f"     Control Weight: {cfg['control_weight']}")
        
        # Assertions
        if cfg['control_model'] != "Test_Model_v1.pth":
            print("[Fail] Model Mismatch!")
        elif cfg['control_module'] != "canny":
            print("[Fail] Module Mismatch!")
        elif cfg['control_weight'] != 0.75:
            print(f"[Fail] Weight Mismatch! Got {cfg['control_weight']}")
        else:
            print("[Pass] ControlNet UI Wiring is CORRECT.")
            
    except Exception as e:
        print(f"[Error] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_controlnet_wiring()
