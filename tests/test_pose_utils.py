import sys
import os
import cv2
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_utils import draw_pose_map

def test_draw_pose_map():
    print("Testing draw_pose_map...")
    
    # Dummy Keypoints (17 points, x,y,conf)
    # Simulate a standing person
    kps = np.random.rand(17, 3) * 512
    # Set high confidence
    kps[:, 2] = 0.9 
    
    h, w = 512, 512
    
    pose_map = draw_pose_map([kps], h, w)
    
    if pose_map.shape != (h, w, 3):
        print(f"FAILED: Shape mismatch {pose_map.shape}")
        return
        
    # Check if not empty
    if np.sum(pose_map) == 0:
        print("FAILED: Pose map is empty (black)")
        return
        
    print(f"SUCCESS: Pose map generated (Shape: {pose_map.shape}, Non-zero pixels: {np.count_nonzero(pose_map)})")
    
    # Optional: Save for manual inspection
    out_path = "test_pose_map.png"
    cv2.imwrite(out_path, pose_map)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    test_draw_pose_map()
