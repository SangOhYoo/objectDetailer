import sys
import os
import cv2
import numpy as np
import torch

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interrogator import Interrogator

def test_interrogator():
    print("=== Testing Interrogator (WD14 Tagger) ===")
    
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    interrogator = Interrogator(device=device)
    
    # 2. Create Dummy Image (or use real one if available)
    # Since we want to test if it actually downloads and runs, 
    # we'll use a random noisey image focused in the center to mimic a 'face' feature
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 100, (200, 200, 200), -1) # "Face"
    
    # 3. Interrogate
    print("Running interrogation (This may trigger download on first run)...")
    try:
        tags = interrogator.interrogate(img, threshold=0.35)
        print(f"Detected Tags: {tags}")
        
        if tags is not None:
            print("SUCCESS: Interrogator returned tags.")
            # Even if empty string (no tags above thresh), it's a success in flow
        else:
            print("FAILED: Interrogator returned None.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_interrogator()
