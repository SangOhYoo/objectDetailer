import os
import torch
import numpy as np
import cv2
import sys

# [Patch] Robust torchvision 0.18+ compatibility for basicsr
try:
    import torchvision.transforms.functional as F
    import sys as sys_mod
    from types import ModuleType
    if 'torchvision.transforms.functional_tensor' not in sys_mod.modules:
        ft_module = ModuleType('torchvision.transforms.functional_tensor')
        ft_module.rgb_to_grayscale = F.rgb_to_grayscale
        if hasattr(F, 'to_tensor'): ft_module.to_tensor = F.to_tensor
        sys_mod.modules['torchvision.transforms.functional_tensor'] = ft_module
except Exception as e:
    print(f"[Upscaler] Patch failed: {e}")


try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("[Upscaler] basicsr not found or failed to import.")
    RRDBNet = None

import torch.nn as nn
import torch.nn.functional as F

class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for Real-ESRGAN.
    Based on Basicsr implementation.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super(SRVGGNetCompact, self).__init__()
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.scale = upscale

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the mid convs
        for _ in range(num_conv):
            self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))

        # the last conv
        self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.pixel_shuffle(out)
        # add the resized input
        if self.scale > 1:
            base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
            out += base
        return out

class ESRGANUpscaler:
    def __init__(self, device='cuda', log_callback=None):
        self.device = device
        self.model = None
        self.current_model_path = None
        self.log_callback = log_callback

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(f"[Upscaler] {msg}")

    def load_model(self, model_path):
        self.log(f"Attempting to load model: {os.path.basename(model_path)}")
        if RRDBNet is None:
            self.log("ERROR: RRDBNet (basicsr) is None. Import failed.")
            return False
        if not os.path.exists(model_path):
            self.log(f"ERROR: Model file not found at: {model_path}")
            return False
            
        if self.current_model_path == model_path and self.model is not None:
             return True
        
        try:
            # 1. Load State Dict
            state_dict = torch.load(model_path, map_location=self.device)
            if 'params_ema' in state_dict: state_dict = state_dict['params_ema']
            elif 'params' in state_dict: state_dict = state_dict['params']
                
            # 2. Detect Scale & Architecture Parameters
            is_compact_4x = False # Flag for AnimeSharp-style models
            is_srvgg = 'body.0.weight' in state_dict and 'conv_first.weight' not in state_dict and 'model.0.weight' not in state_dict
            
            detected_scale = 1
            num_block = 0
            num_feat = 64
            
            if is_srvgg:
                self.log("Detected SRVGGNetCompact architecture.")
                # Detect scale from last conv
                last_key = sorted([k for k in state_dict.keys() if k.startswith('body.') and k.endswith('.weight')], 
                                  key=lambda x: int(x.split('.')[1]))[-1]
                out_ch = state_dict[last_key].shape[0]
                detected_scale = int((out_ch // 3)**0.5)
                # Detect feat
                num_feat = state_dict['body.0.weight'].shape[0]
                # Detect conv count
                # body.0(conv), body.1(prelu), body.2(conv)... body.N-1(prelu), body.N(last_conv)
                # num_keys = (num_conv * 2) + 2 (first + last)
                # Wait, my implementation adds first, then num_conv * [prelu, conv], then prelu, then last_conv.
                # Total convs in body = 1 + num_conv + 1 = num_conv + 2
                max_idx = int(last_key.split('.')[1])
                num_conv = (max_idx - 2) // 2
                
                self.log(f"Detected: {os.path.basename(model_path)} | Scale={detected_scale}x, Feats={num_feat}, MidConvs={num_conv}")
                self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=detected_scale)
            
            else:
                # RRDBNet detection
                has_up1 = 'model.6.weight' in state_dict or 'conv_up1.weight' in state_dict
                has_up2 = 'model.8.weight' in state_dict or 'conv_up2.weight' in state_dict
                
                if has_up1 and has_up2: detected_scale = 4
                elif has_up1: detected_scale = 2
                
                max_body_idx = -1
                for k in state_dict.keys():
                    if 'body' in k or 'model.1.sub.' in k:
                        parts = k.split('.')
                        for p in parts:
                            if p.isdigit():
                                idx = int(p)
                                if idx > max_body_idx: max_body_idx = idx
                                break
                
                if max_body_idx >= 0: num_block = max_body_idx + 1
                else: num_block = 23 # Fallback
                
                if 'model.0.weight' in state_dict:
                    num_feat = state_dict['model.0.weight'].shape[0]
                elif 'conv_first.weight' in state_dict:
                    num_feat = state_dict['conv_first.weight'].shape[0]

                self.log(f"Detected: {os.path.basename(model_path)} | Scale={detected_scale}x, Blocks={num_block}, Feats={num_feat}")

                # 3. Convert State Dict
                state_dict = self._convert_state_dict(state_dict, detected_scale)
                
                if num_block == 24:
                    self.log("Clamping 24 blocks to 23 for compatibility.")
                    num_block = 23

                # 4. Initialize Model
                self.log(f"Initializing RRDBNet(scale={detected_scale}, blocks={num_block}, feats={num_feat})...")
                
                # [Fix] basicsr for scale=1 (and sometimes 2) uses a PixelUnshuffle template
            # [Fix] basicsr for scale=1 (and sometimes 2) uses a PixelUnshuffle template
            # resulting in 48 (3*16) channels in conv_first even if num_in_ch=3 is passed.
            # If our model checkpoint has [feats, 3, 3, 3], we must use a scale that gives 3 channels.
            if not is_srvgg:
                actual_in_ch = state_dict['conv_first.weight'].shape[1]
                init_scale = detected_scale
            else:
                actual_in_ch = 3
                init_scale = detected_scale
            
            # 4. Handle Scale Mismatch (1x, 2x with 4x architecture)
            if not is_srvgg and detected_scale < 3 and actual_in_ch == 3:
                self.log(f"  > Detected size mismatch potential for {detected_scale}x model.")
                self.log(f"  > Using scale=4 template with dynamic monkey-patch.")
                
                self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, 
                                     num_grow_ch=32, scale=4)
                
                model_indices = sorted(list(set([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('model.') and k.split('.')[1].isdigit()])))
                max_idx = max(model_indices) if model_indices else 10
                self.model._antigravity_max_idx = max_idx
                
                # [Diagnostic] Check for activation layer between conv_body and conv_last
                # Indices: 0(first), 1(body), 2(body_conv?), 4(last?)
                # If there's a gap (like 3), it's likely an activation.
                use_lrelu = (max_idx - (max_idx-2)) > 1 # Simple heuristic: gap > 1
                self.log(f"  > Max index detected: {max_idx}. Use LReLU heuristic: {use_lrelu}")

                import types
                def patched_forward(self, x):
                    feat = self.conv_first(x)
                    body_feat = self.body(feat)
                    feat = feat + self.conv_body(body_feat)
                    # 1x models (especially 'Lite' variants) often omit the final LReLU
                    # and expect the Global Residual to be passed directly to conv_last.
                    out = self.conv_last(feat)
                    return out
                
                self.model.forward = types.MethodType(patched_forward, self.model)
                self.log("  > Forward pass monkey-patched successfully (1x Residual).")
            
            # 5. Handle Compact 4x Models (AnimeSharp, etc.)
            # model.10 was mapped to conv_hr (64->64) by _convert_state_dict default logic
            # but for Compact models, it is actually conv_last (64->3).
            elif not is_srvgg and detected_scale == 4 and 'conv_hr.weight' in state_dict and state_dict['conv_hr.weight'].shape[0] == 3:
                self.log(f"  > Detected Compact 4x RRDBNet (e.g., AnimeSharp).")
                self.log(f"  > Monkey-patching to skip conv_hr.")
                
                self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, 
                                     num_grow_ch=32, scale=4)
                
                # Correct the mapping in-place
                if 'conv_hr.weight' in state_dict:
                    state_dict['conv_last.weight'] = state_dict.pop('conv_hr.weight')
                if 'conv_hr.bias' in state_dict:
                    state_dict['conv_last.bias'] = state_dict.pop('conv_hr.bias')
                
                import types
                import torch.nn.functional as F
                def patched_forward(self, x):
                    feat = self.conv_first(x)
                    body_feat = self.conv_body(self.body(feat))
                    feat = feat + body_feat
                    # upsample
                    feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
                    feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
                    # Compact model direct output
                    out = self.conv_last(self.lrelu(feat))
                    return out
                
                self.model.forward = types.MethodType(patched_forward, self.model)
                self.log("  > Forward pass monkey-patched successfully (Compact 4x).")

            else:
                self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, num_grow_ch=32, scale=init_scale)

            # 5. Load Weights
            self.log("Loading state_dict into RRDBNet...")
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.log(f"Moving model to {self.device}...")
            self.model = self.model.to(self.device)
            self.current_model_path = model_path
            
            self.log(f"Model {os.path.basename(model_path)} loaded successfully.")
            return True
            
        except Exception as e:
            self.log(f"FAILED to load {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _convert_state_dict(self, state_dict, scale, is_compact_4x=False):
        """Converts model keys to standard RRDBNet format based on scale."""
        new_state_dict = {}
        
        # Dynamic detection of final layer for 1x/Monkey-Patched models
        max_idx = getattr(self.model, '_antigravity_max_idx', -1)
        
        for k, v in state_dict.items():
            if k.startswith('model.0.'):
                new_k = k.replace('model.0.', 'conv_first.')
            elif k.startswith('model.1.sub.'):
                new_k = k.replace('model.1.sub.', 'body.')
                # BasicSR to RRDBNet mapping
                new_k = new_k.replace('.RDB', '.rdb')
                new_k = new_k.replace('.0.weight', '.weight').replace('.0.bias', '.bias')
            elif max_idx != -1 and k.startswith(f'model.{max_idx}.'):
                # In monkey-patch mode, the very last layer is mapped to conv_last
                new_k = k.replace(f'model.{max_idx}.', 'conv_last.')
            elif k.startswith('model.2.') or k.startswith('model.3.'):
                # Map whatever is here to conv_body
                new_k = k.replace(f'model.{k.split(".")[1]}.', 'conv_body.')
            elif k.startswith('model.6.'):
                new_k = k.replace('model.6.', 'conv_up1.') if scale >= 2 else k
            elif k.startswith('model.8.'):
                new_k = k.replace('model.8.', 'conv_up2.') if scale >= 4 else k
            elif k.startswith('model.10.'):
                # Standard ESRGAN: model.10 is conv_hr
                # Check for Compact 4x override
                if is_compact_4x:
                    new_k = k.replace('model.10.', 'conv_last.')
                else:
                    new_k = k.replace('model.10.', 'conv_hr.')
            elif k.startswith('model.12.'):
                # Standard ESRGAN: model.12 is conv_last
                new_k = k.replace('model.12.', 'conv_last.')
            else:
                new_k = k
            
            new_state_dict[new_k] = v
            
        return new_state_dict

    def upscale(self, img_np, scale_factor=None):
        """
        img_np: HWC, BGR, 0-255, uint8
        return: HWC, BGR, 0-255, uint8
        """
        if self.model is None:
            return img_np
            
        # [VRAM Optimization] Auto-Tiling for large images
        # If any dimension > 512, use tiled upscaling to prevent OOM
        h, w = img_np.shape[:2]
        if h > 512 or w > 512:
            print(f"[Upscaler] Large image detected ({w}x{h}). Using tiled upscaling...")
            return self._upscale_tiled(img_np)

        # Pre-process
        is_1x = getattr(self.model, 'scale', 1) == 1
        if img_np.shape[2] == 3 and not is_1x:
            # Standard ESRGANs are RGB, but many 1x Lite models are BGR-based
            img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img = img_np
            
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # [Diagnostic] Input stats
        if self.current_model_path and "Lite" in self.current_model_path:
            avg_in = np.mean(img_np, axis=(0,1))
            print(f"[Upscaler Debug] Input mean: {avg_in}")

        # Inference
        try:
            with torch.no_grad():
                 output = self.model(img)
        except Exception as e:
            print(f"[Upscaler] Inference Failed: {e}")
            return img_np
             
        # Post-process
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        
        if np.any(np.isnan(output)):
            print("[Upscaler] Warning: NaN detected in output tensor!")
            return img_np
            
        if output.shape[2] == 3 and not is_1x:
             output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        output = (output * 255.0).round().astype(np.uint8)
        
        # [Diagnostic] Output stats
        if self.current_model_path and "Lite" in self.current_model_path:
            avg_out = np.mean(output, axis=(0,1))
            print(f"[Upscaler Debug] Output mean: {avg_out}")

        return output

    def _upscale_tiled(self, img_np, tile_size=400, tile_pad=32):
        """
        Upscales image in Tiles to save VRAM.
        Based on BasicSR/SwinIR tiled inference logic.
        """
        h, w, c = img_np.shape
        # Pre-convert to RGB float tensor for consistency (Unless 1x)
        scale = getattr(self.model, 'scale', 4)
        is_1x = (scale == 1)
        if c == 3 and not is_1x:
             img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
             img_rgb = img_np
             
        img_tensor = torch.from_numpy(np.transpose(img_rgb.astype(np.float32) / 255., (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        b, c, h, w = img_tensor.shape
        tile_size = min(tile_size, h, w)
        
        stride = tile_size - tile_pad
        
        output_shape = (b, c, h * scale, w * scale)
        output = torch.zeros(output_shape).to(self.device)
        output_mask = torch.zeros(output_shape).to(self.device)
        
        # Feathering mask for smooth blending
        mask = torch.ones((1, c, tile_size * scale, tile_size * scale)).to(self.device)
        # Apply linear ramp to edges of the mask
        if tile_pad > 0:
            pad_s = tile_pad * scale
            for i in range(pad_s):
                mask[:, :, i, :] *= (i / pad_s)
                mask[:, :, -i-1, :] *= (i / pad_s)
                mask[:, :, :, i] *= (i / pad_s)
                mask[:, :, :, -i-1] *= (i / pad_s)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Extract tile
                y_start, x_start = y, x
                y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
                
                # Adjust if tile is smaller than tile_size at edges
                if y_end - y_start < tile_size: y_start = max(0, y_end - tile_size)
                if x_end - x_start < tile_size: x_start = max(0, x_end - tile_size)
                
                tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
                
                # Upscale tile
                with torch.no_grad():
                    tile_output = self.model(tile)
                
                # Accumulate
                output[:, :, y_start*scale:y_end*scale, x_start*scale:x_end*scale] += tile_output * mask
                output_mask[:, :, y_start*scale:y_end*scale, x_start*scale:x_end*scale] += mask
                
        # Final Blend
        output = (output / output_mask).clamp(0, 1)
        
        # Convert back to HWC BGR uint8
        output_np = output.data.squeeze().float().cpu().numpy()
        output_np = np.transpose(output_np, (1, 2, 0))
        if c == 3 and not is_1x:
             output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        else:
             output_bgr = output_np
        return (output_bgr * 255.0).round().astype(np.uint8)
