import torch
import os

model_path = r"D:/AI_Models/ESRGAN/4x-AnimeSharp.pth"
if os.path.exists(model_path):
    sd = torch.load(model_path, map_location='cpu')
    if 'params' in sd: sd = sd['params']
    
    keys = sorted(sd.keys())
    print(f"Total keys: {len(keys)}")
    
    # Filter for high-level structure
    print("--- Top Level Layers ---")
    for k in keys:
        if k.endswith('.weight'):
            if "model.1.sub" not in k:
                print(f"{k} | {sd[k].shape}")
            elif "model.1.sub.0" in k: # Show just one block as example
                 print(f"{k} | {sd[k].shape}")
                 
    # Check max index
    indices = []
    for k in keys:
        parts = k.split('.')
        if len(parts) > 1 and parts[1].isdigit():
            indices.append(int(parts[1]))
    if indices:
        print(f"Max model index: {max(indices)}")

else:
    print(f"Model not found at {model_path}")
