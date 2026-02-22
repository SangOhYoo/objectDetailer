import torch
import os

model_path = r"D:/AI_Models/ESRGAN/x1_ITF_SkinDiffDetail_Lite_v1.pth"
if os.path.exists(model_path):
    sd = torch.load(model_path, map_location='cpu')
    keys = sorted(sd.keys())
    print(f"Total keys: {len(keys)}")
    print("--- ALL KEYS ---")
    for k in keys:
        if 'model.' in k:
            print(f"{k} | {sd[k].shape}")
else:
    print(f"Model not found at {model_path}")
