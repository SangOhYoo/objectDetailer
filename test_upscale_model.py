import torch
import cv2
import numpy as np
import os
import sys
import types

# Add current dir to path for imports
sys.path.append(os.getcwd())

from core.upscaler import ESRGANUpscaler, RRDBNet

def test_variants():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = r"D:/AI_Models/ESRGAN/x1_ITF_SkinDiffDetail_Lite_v1.pth"
    sd = torch.load(model_path, map_location='cpu')
    
    # Blocks and feats detection
    blocks = 0
    for k in sd.keys():
        if k.startswith('model.1.sub.'):
            parts = k.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                blocks = max(blocks, int(parts[3]) + 1)
    
    # Wait, the keys are model.1.sub.0.... or model.1.sub.12....
    # Let's re-examine keys from probe: ['model.1.sub.0.RDB1.conv1.0.weight', ...]
    # So k.split('.') is ['model', '1', 'sub', '0', 'RDB1', ...]
    # parts[3] is indeed the index.
    
    feats = sd['model.0.weight'].shape[0]
    print(f"Detected: Blocks={blocks}, Feats={feats}")

    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_img[:, :] = [0, 0, 255] # BGR Pure Red
    
    variants = [
        ("LReLU_True", True),
        ("LReLU_False", False)
    ]
    
    for name, use_lrelu in variants:
        print(f"\n--- Testing variant: {name} ---")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=feats, num_block=blocks, 
                             num_grow_ch=32, scale=4)
        
        # Mapping
        new_sd = {}
        max_idx = 4
        for k, v in sd.items():
            if k.startswith('model.0.'): new_k = k.replace('model.0.', 'conv_first.')
            elif k.startswith('model.1.sub.'): 
                new_k = k.replace('model.1.sub.', 'body.')
                new_k = new_k.replace('.RDB', '.rdb').replace('.0.weight', '.weight').replace('.0.bias', '.bias')
            elif k.startswith('model.2.'): new_k = k.replace('model.2.', 'conv_body.')
            elif k.startswith('model.4.'): new_k = k.replace('model.4.', 'conv_last.')
            else: new_k = k
            new_sd[new_k] = v
        
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        
        def patched_forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.body(feat)
            res = self.conv_body(body_feat)
            feat = feat + res
            if use_lrelu:
                return self.conv_last(self.lrelu(feat))
            else:
                return self.conv_last(feat)
        
        model.forward = types.MethodType(patched_forward, model)
        
        # Test 1: RGB Input (Swapped)
        img_rgb = test_img[:, :, ::-1].copy()
        t_in = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.
        with torch.no_grad():
            out_rgb = model(t_in).clamp(0, 1).squeeze().numpy().transpose(1, 2, 0)
        out_bgr_test1 = (out_rgb[:, :, ::-1] * 255).astype(np.uint8)
        print(f"  RGB Input -> BGR Output means: {np.mean(out_bgr_test1, axis=(0,1))}")
        
        # Test 2: BGR Input (Direct)
        t_in = torch.from_numpy(test_img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.
        with torch.no_grad():
            out_bgr_test2 = model(t_in).clamp(0, 1).squeeze().numpy().transpose(1, 2, 0)
        out_bgr_test2 = (out_bgr_test2 * 255).astype(np.uint8)
        print(f"  BGR Input -> BGR Output means: {np.mean(out_bgr_test2, axis=(0,1))}")

if __name__ == "__main__":
    test_variants()
