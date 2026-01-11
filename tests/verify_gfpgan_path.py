import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

def test_gfpgan_path_logic():
    from core.face_restorer import FaceRestorer
    import core.config
    
    # 1. Setup mocks
    mock_cfg = MagicMock()
    core.config.config_instance = mock_cfg
    mock_cfg.get_path.return_value = "C:/fallback/gfpgan"
    
    restorer = FaceRestorer(device='cpu')
    
    # We want to check if it tries to load from D:/AI_Models/GFPGAN first.
    # We'll mock os.path.exists to simulate the file existing at the new path.
    
    fixed_path = os.path.normpath("D:/AI_Models/GFPGAN/GFPGANv1.4.pth")
    
    with patch('os.path.exists') as mock_exists:
        with patch('gfpgan.GFPGANer') as mock_gfpganer:
            # Simulate GFPGANv1.4.pth exists at D:/AI_Models/GFPGAN
            mock_exists.side_effect = lambda p: os.path.normpath(p) == fixed_path
            
            success = restorer.load_model()
            
            print(f"Load Success: {success}")
            print(f"FACEXLIB_HOME: {os.environ.get('FACEXLIB_HOME')}")
            
            # Verify GFPGANer was called with the correct path
            args, kwargs = mock_gfpganer.call_args
            actual_path = kwargs.get('model_path')
            print(f"Actual model path passed: {actual_path}")
            
            assert actual_path == fixed_path, f"Path mismatch: {actual_path}"
            assert os.environ.get('FACEXLIB_HOME') == "D:/AI_Models/GFPGAN", "FACEXLIB_HOME mismatch"
            
            print("\n[NOTE] facexlib typically expects models in {FACEXLIB_HOME}/weights/")
            print("If models are directly in D:/AI_Models/GFPGAN/, you may need to rename GFPGAN to 'weights'")
            print("or create a 'weights' subfolder inside it.")
            
            if success:
                print("\n[VERIFICATION] SUCCESS: GFPGAN logic correctly prioritizes D:/AI_Models/GFPGAN")
            else:
                print("\n[VERIFICATION] FAILED: load_model returned False")

if __name__ == "__main__":
    try:
        test_gfpgan_path_logic()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
