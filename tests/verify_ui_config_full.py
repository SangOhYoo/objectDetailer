
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
        if 'control_weight' in widget.settings:
            spin = widget.settings['control_weight']['widget']
            spin.setValue(0.75) 
            print(f"     Set Weight SpinBox to 0.75")
        else:
            print("[Fail] 'control_weight' widget not found in settings!")
            
        # 2. ComboBox: Model
        widget.combo_cn_model.addItem("Test_Model_v1.pth")
        widget.combo_cn_model.setCurrentText("Test_Model_v1.pth")
        print(f"     Set Model to 'Test_Model_v1.pth'")
        
        # 3. ComboBox: Module (Preprocessor)
        widget.combo_cn_module.setCurrentText("canny")
        print(f"     Set Preprocessor to 'canny'")
        
        # 3. Modify UI Elements
        print("[Test] Setting UI Elements...")
        
        # --- Soft Inpainting ---
        print("   -> Testing Soft Inpainting...")
        if hasattr(widget, 'g_soft'):
            widget.g_soft.setChecked(True)
            # Sliders (SpinBoxes)
            widget.settings['soft_schedule_bias']['widget'].setValue(2.5)
            widget.settings['soft_preservation_strength']['widget'].setValue(0.75)
            widget.settings['soft_transition_contrast']['widget'].setValue(10.0)
            widget.settings['soft_mask_influence']['widget'].setValue(0.25)
            widget.settings['soft_diff_threshold']['widget'].setValue(0.15)
            widget.settings['soft_diff_contrast']['widget'].setValue(1.5)
        else:
            print("[Fail] g_soft groupbox not found!")

        # --- Mask Content ---
        print("   -> Testing Mask Content...")
        widget.radio_content_noise.setChecked(True)
        
        # --- Inpaint Area ---
        print("   -> Testing Inpaint Area...")
        widget.radio_area_whole.setChecked(True)
        
        # --- Landscape Detail ---
        print("   -> Testing Landscape Detail...")
        widget.chk_landscape_detail.setChecked(True)
        
        # --- BMAB Edge ---
        print("   -> Testing BMAB Edge...")
        if hasattr(widget, 'g_edge'):
            widget.g_edge.setChecked(True)
            widget.settings['bmab_edge_strength']['widget'].setValue(0.8)
        else:
            print("[Fail] g_edge groupbox not found!")

        # 4. Get Config
        cfg = widget.get_config()
        print("\n[Test] Retrieved Config:")
        
        # 5. Verify
        failures = []
        
        # Function to check assertion
        def check(key, expected, tolerance=0.001):
            if key not in cfg:
                failures.append(f"Missing key: {key}")
                return
            val = cfg[key]
            if isinstance(expected, float):
                if abs(val - expected) > tolerance:
                    failures.append(f"Key '{key}': Expected {expected}, Got {val}")
            else:
                if val != expected:
                    failures.append(f"Key '{key}': Expected {expected}, Got {val}")

        # Checks
        check('control_model', "Test_Model_v1.pth")
        check('control_module', "canny")
        check('control_weight', 0.75)

        check('use_soft_inpainting', True)
        check('soft_schedule_bias', 2.5)
        check('soft_preservation_strength', 0.75)
        check('soft_transition_contrast', 10.0)
        check('soft_mask_influence', 0.25)
        check('soft_diff_threshold', 0.15)
        check('soft_diff_contrast', 1.5)
        
        check('mask_content', 'latent_noise')
        check('inpaint_full_res', True)
        check('bmab_landscape_detail', True)
        
        check('bmab_edge_enabled', True)
        check('bmab_edge_strength', 0.8)

        if failures:
            print("[Fail] Verification Failed with errors:")
            for f in failures:
                print(f"  - {f}")
        else:
            print("[Pass] ALL UI Features Verified Successfully!")
            
    except Exception as e:
        print(f"[Error] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_controlnet_wiring()
