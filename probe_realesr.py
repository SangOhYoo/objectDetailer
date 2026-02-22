import torch
import os

model_path = r"D:/AI_Models/ESRGAN/realesr-general-wdn-x4v3.pth"
if os.path.exists(model_path):
    sd = torch.load(model_path, map_location='cpu')
    if 'params' in sd:
        inner = sd['params']
        keys = sorted(inner.keys(), key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 999)
        print(f"Total inner keys: {len(keys)}")
        for k in keys:
            print(f"{k} | {inner[k].shape}")
else:
    print(f"Model not found at {model_path}")
