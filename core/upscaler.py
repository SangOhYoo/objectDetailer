import os
import torch
import numpy as np
import cv2
import sys

# [Patch] Fix torchvision 0.18+ compatibility for basicsr
try:
    import torchvision.transforms.functional as F
    if not hasattr(F, 'rgb_to_grayscale'):
        F.rgb_to_grayscale = lambda x, num_output_channels=1: F.convert_image_dtype(x, torch.float32).mean(dim=-3, keepdim=True)
    # Also patch functional_tensor if needed by importing it and setting it? 
    # basicsr checks torchvision.transforms.functional_tensor
    import torchvision.transforms
    if not hasattr(torchvision.transforms, 'functional_tensor'):
        # Create a dummy module or alias
        # But basicsr imports FROM it.
        # Check where basicsr fails: "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        # We can simulate this module in sys.modules
        pass
except ImportError:
    pass

# Patch sys.modules to prevent ModuleNotFoundError for functional_tensor
# This is a hack because basicsr imports strictly from there
try:
    import torchvision.transforms.functional as F
    import sys
    from types import ModuleType
    
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        ft_module = ModuleType('torchvision.transforms.functional_tensor')
        ft_module.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = ft_module
except Exception as e:
    print(f"[Upscaler] Patch failed: {e}")


# Now import basicsr
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("[Upscaler] basicsr not found or failed to import.")
    RRDBNet = None

class ESRGANUpscaler:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.current_model_path = None

    def load_model(self, model_path):
        if RRDBNet is None:
            print("[Upscaler] RRDBNet not available.")
            return False
        if not os.path.exists(model_path):
            print(f"[Upscaler] Model not found: {model_path}")
            return False
            
        if self.current_model_path == model_path and self.model is not None:
             return True
        
        try:
            # 1. Load State Dict first to inspect architecture
            # [Fix] Use weights_only=False due to legacy models, but user is aware.
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle standard keys vs flat keys (wrapped in 'params' or 'params_ema')
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
                
            # Convert keys if necessary (Old ESRGAN -> RRDBNet)
            state_dict = self._convert_state_dict(state_dict)
            
            # 2. Detect Architecture Parameters from keys
            # num_block: max index of 'body.X'
            # num_feat: shape of 'conv_first.weight' [Out, In, k, k] -> Out is num_feat
            
            num_block = 0
            num_feat = 64 # Default
            max_body_idx = -1
            
            for k in state_dict.keys():
                if k.startswith('body.'):
                    # expected format: body.22.rdb3...
                    parts = k.split('.')
                    if len(parts) > 1 and parts[1].isdigit():
                        idx = int(parts[1])
                        if idx > max_body_idx:
                            max_body_idx = idx
            
            if max_body_idx >= 0:
                num_block = max_body_idx + 1
            else:
                # Fallback: simple models might not have body.X?
                # or maybe just conv_first/last?
                # Assume standard 23 if not found but keys look valid?
                # If we converted correctly, body.X should exist for RRDBNet
                pass
                
            if 'conv_first.weight' in state_dict:
                # Shape: [Out, In, K, K]
                nf = state_dict['conv_first.weight'].shape[0]
                num_feat = nf
                
            print(f"[Upscaler] Detected params from {os.path.basename(model_path)}: num_block={num_block}, num_feat={num_feat}")
            
            if num_block == 0:
                print(f"[Upscaler] Warning: Could not detect blocks. Assuming default RRDBNet(nb=23).")
                num_block = 23
            elif num_block == 24:
                # [Fix] 4x-UltraSharp contains 'sub.23' (24 blocks) keys but behaves like standard ESRGAN (23 blocks)?
                # Random initialization of the 24th block causes "strange" artifacts.
                # Force clamp to 23.
                print(f"[Upscaler] Detected 24 blocks. Forcing 23 to match standard ESRGAN (ignoring last block).")
                num_block = 23
                
            if num_feat != 64:
                 print(f"[Upscaler] Note: Non-standard num_feat={num_feat}")

            # 3. Initialize Model with detected params
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=32, scale=4)
            
            # 4. Load Weights
            # use strict=False to allow for minor mismatch (e.g. conv_hr injection)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            self.current_model_path = model_path
            
            return True
            
        except Exception as e:
            print(f"[Upscaler] Failed to load {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _convert_state_dict(self, state_dict):
        """Converts 'model.X' style keys to basicsr RRDBNet 'conv_first'/'body' style."""
        new_state_dict = {}
        
        # [Fix] Detect if model.10 is conv_last (Out=3)
        # Check model.10.weight shape if it exists
        model_10_is_last = False
        if 'model.10.weight' in state_dict:
            shp = state_dict['model.10.weight'].shape
            # shp is [Out, In, k, k]
            if shp[0] == 3:
                model_10_is_last = True
                print("[Upscaler] Detected model.10 as Output Layer (Skipping conv_hr)")
        
        for k, v in state_dict.items():
            if k.startswith('model.0.'):
                new_k = k.replace('model.0.', 'conv_first.')
            elif k.startswith('model.1.sub.'):
                # model.1.sub.X.RDBY.convZ -> body.X.rdbY.convZ
                new_k = k.replace('model.1.sub.', 'body.')
                new_k = new_k.replace('.RDB', '.rdb')
                new_k = new_k.replace('.0.weight', '.weight').replace('.0.bias', '.bias')
            elif k.startswith('model.3.'):
                new_k = k.replace('model.3.', 'conv_body.')
            elif k.startswith('model.6.'):
                new_k = k.replace('model.6.', 'conv_up1.')
            elif k.startswith('model.8.'):
                new_k = k.replace('model.8.', 'conv_up2.')
            elif k.startswith('model.10.'):
                if model_10_is_last:
                    new_k = k.replace('model.10.', 'conv_last.')
                else:
                    new_k = k.replace('model.10.', 'conv_hr.')
            elif k.startswith('model.12.'):
                 if model_10_is_last:
                     # If model.10 is last, model.12 shouldn't exist ideally.
                     # But if it does, ignore or map? 
                     # Let's map to nothing/ignore if we decided logic above.
                     new_k = k  
                 else:
                     new_k = k.replace('model.12.', 'conv_last.')
            else:
                 new_k = k
            
            new_state_dict[new_k] = v

        # [Fix] Helper: Create Identity Weights for missing conv_hr
        if model_10_is_last:
            # Inject identity for conv_hr
            # conv_hr: 64 -> 64, 3x3
            # We need to manually add this to state_dict so load_state_dict finds it.
            # Assuming num_feat=64.
            num_feat = 64
            
            # Weight: Identity convolution
            # [64, 64, 3, 3]
            kw = torch.zeros((num_feat, num_feat, 3, 3), dtype=torch.float32)
            for i in range(num_feat):
                kw[i, i, 1, 1] = 1.0
            
            new_state_dict['conv_hr.weight'] = kw
            new_state_dict['conv_hr.bias'] = torch.zeros(num_feat, dtype=torch.float32)
            
        return new_state_dict

    def upscale(self, img_np, scale_factor=None):
        """
        img_np: HWC, BGR, 0-255, uint8
        return: HWC, BGR, 0-255, uint8
        """
        if self.model is None:
            return img_np
            
        # Pre-process
        # BasicSR / Spandrel Standard: Feed RGB tensor to model
        # Input img_np is BGR (OpenCV) -> Convert to RGB
        # [Fix] Hires Fix Color Issue: Models expect RGB, but were receiving BGR.
        if img_np.shape[2] == 3:
            img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img = img_np # Grayscale?
            
        img = img.astype(np.float32) / 255.
        # Transpose HWC -> CHW
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # Inference
        try:
            with torch.no_grad():
                 output = self.model(img)
        except Exception as e:
            print(f"[Upscaler] Inference Failed: {e}")
            return img_np
             
        # Post-process
        # Output is RGB (CHW) -> Convert to HWC numpy -> BGR
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # Transpose to HWC (RGB)
        output = np.transpose(output, (1, 2, 0))
        
        # Convert RGB to BGR
        # [Fix] Handle output color space
        if output.shape[2] == 3:
             output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Convert BGR float to BGR uint8
        output = (output * 255.0).round().astype(np.uint8)
        
        return output
