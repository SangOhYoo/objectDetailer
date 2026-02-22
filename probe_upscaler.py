
import os
import sys
import torch
from types import ModuleType

# [Patch] Robust torchvision 0.18+ compatibility for basicsr
import torchvision.transforms.functional as F
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    ft_module = ModuleType('torchvision.transforms.functional_tensor')
    ft_module.rgb_to_grayscale = F.rgb_to_grayscale
    if hasattr(F, 'to_tensor'): ft_module.to_tensor = F.to_tensor
    sys.modules['torchvision.transforms.functional_tensor'] = ft_module

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print("basicsr.RRDBNet imported successfully.")
    
    import inspect
    print(f"\nRRDBNet Signature: {inspect.signature(RRDBNet)}")
    
    def probe_model(scale, blocks, feats, in_ch=3):
        print(f"\nProbing RRDBNet(scale={scale}, num_block={blocks}, num_feat={feats}, num_in_ch={in_ch}, num_out_ch=3)")
        try:
            model = RRDBNet(num_in_ch=in_ch, num_out_ch=3, num_feat=feats, num_block=blocks, num_grow_ch=32, scale=scale)
            print(f"  Model created. conv_first weight shape: {model.conv_first.weight.shape}")
        except Exception as e:
            print(f"  Failed: {e}")

    print("\n--- Strategy Check: Monkey-patch forward to skip upsampling ---")
    def probe_patch():
        print(f"Initializing RRDBNet(scale=4, num_in_ch=3)")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=2, num_grow_ch=32, scale=4)
        
        # Original forward might be complex. Let's see if we can just re-def it.
        def patched_forward(self, x):
            # Typical RRDBNet forward simplified for 1x
            feat = self.conv_first(x)
            body_feat = self.body(feat)
            body_feat = self.conv_body(body_feat)
            feat = feat + body_feat
            # Skip conv_up1, conv_up2, etc.
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

        import types
        model.forward = types.MethodType(patched_forward, model)
        print("  Model patched.")
        
        x = torch.randn(1, 3, 64, 64)
        try:
            out = model(x)
            print(f"  Forward output shape: {out.shape}")
        except Exception as e:
            print(f"  Forward failed: {e}")
            import traceback
            traceback.print_exc()

    probe_patch()
    
except ImportError:
    print("basicsr not found.")
except Exception as e:
    print(f"Error: {e}")
