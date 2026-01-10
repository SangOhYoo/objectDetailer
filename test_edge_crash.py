
import numpy as np
import cv2

def test_edge_enhancement():
    # Mock image (H, W, 3)
    h, w = 512, 512
    proc_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw some white circle
    cv2.circle(proc_img, (256, 256), 100, (255, 255, 255), -1)
    
    config = {
        'bmab_edge_strength': 1.0,
        'bmab_edge_low': 50,
        'bmab_edge_high': 200
    }
    
    # Logic from pipeline.py
    try:
        low = config.get('bmab_edge_low', 50)
        high = config.get('bmab_edge_high', 200)
        strength = config['bmab_edge_strength']
        
        # Detect Edges
        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        print(f"Edges shape: {edges.shape}, dtype: {edges.dtype}")
        
        # Blend Edges
        proc_img_f = proc_img.astype(np.float32)
        mask = edges > 0
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        
        # THIS IS THE SUSPECT LINE
        proc_img_f[mask] -= (255.0 * strength)
        
        proc_img = np.clip(proc_img_f, 0, 255).astype(np.uint8)
        print("Success!")
        
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_edge_enhancement()
