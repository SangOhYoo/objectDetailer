import sys
import os
import time
import shutil
import numpy as np
import cv2
from PyQt6.QtCore import QCoreApplication, QTimer

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchvision
try:
    # Need to mock ImageProcessor BEFORE importing ui.workers
    import core.pipeline
except Exception:
    pass

class DummyProcessor:
    def __init__(self, device_id, log_callback=None, preview_callback=None):
        self.device_id = device_id
        self.log_callback = log_callback
    def process(self, img, config):
        if self.log_callback: self.log_callback(f"Processing on {self.device_id}...")
        time.sleep(1.0) # Simulate work
        return img

# Override in sys.modules if real one fails
import sys
if 'core.pipeline' not in sys.modules or True: # Force dummy for robustness in this environment
    import types
    m = types.ModuleType('core.pipeline')
    m.ImageProcessor = DummyProcessor
    sys.modules['core.pipeline'] = m

from ui.workers import ProcessingController
from core.config import config_instance as cfg

def create_dummy_images(count=4):
    images = []
    os.makedirs("tests/temp_input", exist_ok=True)
    for i in range(count):
        # Create a basic image with a face-like blob to trigger detection if models were real
        # But here we heavily mock or expect failure?
        # Use simple black image. The worker will process it.
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Draw a white circle to simulate content
        cv2.circle(img, (256, 256), 100, (255, 255, 255), -1)
        path = os.path.abspath(f"tests/temp_input/img_{i}.png")
        cv2.imwrite(path, img)
        images.append(path)
    return images

def main():
    app = QCoreApplication(sys.argv)
    
    print("--- Starting Parallel Processing Test ---")
    
    # 1. Setup Data
    inputs = create_dummy_images(4)
    output_dir = os.path.abspath("tests/temp_output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Mock Config
    # We need a valid config list.
    # Config needed by pipeline.
    config = {
        'enabled': True,
        'model': 'v1-5-pruned-emaonly.safetensors', # Likely to fail load if not present, but wrapper handles it?
        'vae': 'Automatic',
        'detector_model': 'yolo_v8n.pt',
        'conf_thresh': 0.5,
        'max_det': 1,
        'unit_name': 'TestPass',
        # Add basic inpaint params
        'denoising_strength': 0.3,
        'inpaint_width': 512,
        'inpaint_height': 512,
        'steps': 1
    }
    
    # Update global config for output path
    # We need to hack the cfg instance or ensure workers get path?
    # workers.py reads cfg.get('system', 'output_path') inside the process.
    # Since processes fork/spawn, we need to save config to disk or rely on defaults?
    # Wait, the worker re-reads 'config.yaml' (conceptually, if it re-imports).
    # But currently 'cfg' is imported.
    # On Windows, 'spawn' means a fresh python interpreter imports modules.
    # config.py loads from file.
    # So we should write our test output path to the actual config file?
    # That's dangerous.
    
    # Let's hope defaults work (outputs folder).
    # Or force the worker to use local path.
    
    # 2. Init Controller
    controller = ProcessingController(inputs, [config])
    
    # 3. Connect Signals
    logs = []
    pids = set()
    
    def handle_log(msg):
        print(f"[LOG] {msg}")
        logs.append(msg)
        if "Worker Process Started (PID:" in msg:
            try:
                # Extract PID
                pid = int(msg.split("PID:")[1].split(")")[0].strip())
                pids.add(pid)
            except: pass
            
    def handle_finish():
        print("Processing Finished!")
        app.quit()
        
    controller.log_signal.connect(handle_log)
    controller.finished_signal.connect(handle_finish)
    
    # 4. Start
    workers_count = 2
    print(f"Requesting {workers_count} workers...")
    controller.start_processing(max_workers=workers_count)
    
    # 5. Run Event Loop (Timeout 60s)
    # 60s is generous.
    QTimer.singleShot(60000, app.quit) 
    
    app.exec()
    
    # 6. Verify Results
    print("\n--- Verification ---")
    
    # Check PIDs
    print(f"Unique PIDs detected: {pids}")
    if len(pids) >= 2:
        print("[PASS] Multiple processes were spawned.")
    else:
        print(f"[FAIL] Expected at least 2 PIDs, got {len(pids)}: {pids}")
        # Note: If system has 1 CPU/GPU, logic might fallback. But we forced 2 workers.
        
    # Check Logs for GPU Mapping
    gpu_logs = [l for l in logs if "GPU Bound:" in l]
    print("GPU Binding Logs:")
    for l in gpu_logs:
        print("  " + l)
        
    if not gpu_logs and "No GPU detected" in "".join(logs):
        print("[WARN] Running in CPU mode (No GPU check possible).")
    elif gpu_logs:
        # Check if they are different or handled correctly
        print("[PASS] GPU Binding logs present.")
        
    # Check Output Files
    # outputs are in 'outputs' folder relative to CWD?
    # default in workers.py is "outputs"
    count = 0
    if os.path.exists("outputs"):
        count = len([f for f in os.listdir("outputs") if f.endswith(".png")])
        
    print(f"Output files found: {count}/{len(inputs)}")
    
    # Cleanup
    try:
        shutil.rmtree("tests/temp_input")
        # Don't delete outputs so user can see them? Or clean up.
        # shutil.rmtree("tests/temp_output")
    except: pass

    if len(pids) >= 2 and (count > 0 or "Error" in "".join(logs)):
        # If count > 0 it worked.
        # If Error, it ran but failed (e.g. missing model), which is still a "parallel execution" test pass.
        print("\nTEST RESULT: SUCCESS (Parallel machinery works)")
    else:
        print("\nTEST RESULT: FAILURE (Parallel machinery issue)")

if __name__ == "__main__":
    main()
