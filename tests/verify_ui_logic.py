
import sys
import os
import yaml
from PyQt6.QtWidgets import QApplication

# Adjust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Config (if needed, but main window loads config.yaml)
# We will trust it loads the actual config.yaml or defaults.

def test_ui_init():
    print("[Test] Initializing QApplication...")
    app = QApplication(sys.argv)
    
    print("[Test] Importing MainWindow...")
    try:
        from ui.main_window import MainWindow
        from ui.main_window_tabs import AdetailerUnitWidget
    except ImportError as e:
        print(f"[Error] Import failed: {e}")
        return

    print("[Test] Creating MainWindow...")
    try:
        window = MainWindow()
        print("[Test] MainWindow created successfully.")
    except Exception as e:
        print(f"[Error] MainWindow init failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Access the first tab (AdetailerUnitWidget)
    try:
        # MainWindow has 'tabs' widget.
        # Assuming tabs are added to self.tab_widget or similar.
        # In init_ui of MainWindow:
        # self.tabs = QTabWidget()
        # for i in range(self.unit_count): tab = AdetailerUnitWidget(...)
        
        # Let's verify we can find the Soft Inpainting checkbox in the first unit
        # MainWindow usually stores units? Or just tabs.
        # Let's inspect window.tabs (QTabWidget)
        
        tab_widget = window.findChild(AdetailerUnitWidget) # Might find first one?
        # Actually `AdetailerUnitWidget` is a QWidget subclass.
        # Let's iterate tabs.
        
        if not tab_widget:
            # Maybe accessed via window.tabs.widget(0)?
            tab_widget = window.tabs.widget(0)
            
        print(f"[Test] Found Tab Widget: {type(tab_widget)}")
        
        if isinstance(tab_widget, AdetailerUnitWidget):
            print("[Test] Verifying Soft Inpainting Checkbox...")
            # g_soft is local var in init_ui, but we can find via header or config check
            # We implemented `get_config` to check `findChild(QGroupBox, "Soft Inpainting")`
            
            cfg = tab_widget.get_config()
            print(f"[Test] Config Retrieved. Keys: {len(cfg)}")
            
            # Check Keys
            required_keys = [
                'mask_content', 'inpaint_full_res', 'use_soft_inpainting',
                'soft_schedule_bias', 'soft_preservation_strength'
            ]
            
            missing = [k for k in required_keys if k not in cfg]
            
            if missing:
                print(f"[Fail] Missing Config Keys: {missing}")
            else:
                print("[Pass] All new config keys present.")
                print(f"       Mask Content: {cfg['mask_content']}")
                print(f"       Soft Inpaint: {cfg['use_soft_inpainting']}")
                
    except Exception as e:
        print(f"[Error] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ui_init()
