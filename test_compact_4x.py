import torch
import cv2
import numpy as np
import os
import sys

# Add current dir to path for imports
sys.path.append(os.getcwd())

from core.upscaler import ESRGANUpscaler

def test_compact_4x():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    upscaler = ESRGANUpscaler(device, log_callback=print)
    
    model_path = r"D:/AI_Models/ESRGAN/4x-AnimeSharp.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print("--- Loading Model ---")
    if not upscaler.load_model(model_path):
        print("Failed to load model")
        return
    
    print("--- Running Inference ---")
    # Test patch
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Gradient test image
    for i in range(100):
        test_img[i, :, :] = i * 2
        
    out = upscaler.upscale(test_img)
    
    print(f"Input shape: {test_img.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output mean: {np.mean(out)}")
    
    if out.shape[0] == 400 and out.shape[1] == 400:
        print("Success: Output is 4x scaled.")
    else:
        print("Failure: Incorrect scaling.")

if __name__ == "__main__":
    test_compact_4x()
