import sys
import os
import torch
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.upscaler import ESRGANUpscaler

def test_upscaler_logic():
    print("=== Testing ESRGANUpscaler Logic ===")
    upscaler = ESRGANUpscaler(device='cpu') # Use CPU for testing logic
    
    # Simulate a 1x model state dict
    # 1x models often lack model.6 (up1) and model.8 (up2)
    state_dict_1x = {
        'model.0.weight': torch.randn(64, 3, 3, 3),
        'model.1.sub.0.rdb1.conv1.weight': torch.randn(64, 64, 3, 3),
        'model.3.weight': torch.randn(64, 64, 3, 3),
        'model.10.weight': torch.randn(3, 64, 3, 3), # Final layer Out=3
        'model.10.bias': torch.randn(3)
    }
    
    # Save dummy model
    dummy_path_1x = "dummy_1x.pth"
    torch.save(state_dict_1x, dummy_path_1x)
    
    try:
        print(f"Loading simulated 1x model...")
        success = upscaler.load_model(dummy_path_1x)
        print(f"Load Success: {success}")
        if success:
            print(f"Detected Scale: {upscaler.model.scale}")
            # Verify if it's 1
            assert upscaler.model.scale == 1, f"Expected scale 1, got {upscaler.model.scale}"
            
            # Test inference dummy
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            out = upscaler.upscale(img)
            print(f"Output shape: {out.shape}")
            assert out.shape == (64, 64, 3), "Output shape should match for 1x"
            
    finally:
        if os.path.exists(dummy_path_1x): os.remove(dummy_path_1x)

    # Simulate a 4x model
    state_dict_4x = {
        'model.0.weight': torch.randn(64, 3, 3, 3),
        'model.1.sub.0.rdb1.conv1.weight': torch.randn(64, 64, 3, 3),
        'model.3.weight': torch.randn(64, 64, 3, 3),
        'model.6.weight': torch.randn(64, 64, 3, 3), # Up1
        'model.8.weight': torch.randn(64, 64, 3, 3), # Up2
        'model.10.weight': torch.randn(64, 64, 3, 3), # HR
        'model.12.weight': torch.randn(3, 64, 3, 3), # Last
        'model.12.bias': torch.randn(3)
    }
    
    dummy_path_4x = "dummy_4x.pth"
    torch.save(state_dict_4x, dummy_path_4x)
    
    try:
        print(f"\nLoading simulated 4x model...")
        success = upscaler.load_model(dummy_path_4x)
        print(f"Load Success: {success}")
        if success:
            print(f"Detected Scale: {upscaler.model.scale}")
            assert upscaler.model.scale == 4
            
            # Test inference dummy
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            out = upscaler.upscale(img)
            print(f"Output shape: {out.shape}")
            assert out.shape == (256, 256, 3), "Output shape should be 4x"

    finally:
        if os.path.exists(dummy_path_4x): os.remove(dummy_path_4x)

if __name__ == "__main__":
    test_upscaler_logic()
