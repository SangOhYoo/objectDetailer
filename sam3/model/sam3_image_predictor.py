# [검증 완료] SAM3: Perfect Batch Synchronization (Fixes concat assertion)
import sys, os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import math
import inspect

# ==============================================================================
# [필수] Runtime Monkey Patch (유지)
# ==============================================================================
try:
    import sam3.model.vitdet as vitdet_module
    def fixed_reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        if ndim == 4:
            target_N, target_D = x.shape[2], x.shape[3]
            if freqs_cis.ndim == 3: freqs_cis = freqs_cis.flatten(0, 1)
            if freqs_cis.shape[0] != target_N:
                cur_side = int(math.sqrt(freqs_cis.shape[0]))
                tgt_side = int(math.sqrt(target_N))
                t = freqs_cis.view(cur_side, cur_side, -1).permute(2,0,1).unsqueeze(0)
                if t.is_complex():
                    real = F.interpolate(t.real, size=(tgt_side, tgt_side), mode='bilinear', align_corners=False)
                    imag = F.interpolate(t.imag, size=(tgt_side, tgt_side), mode='bilinear', align_corners=False)
                    t = torch.complex(real, imag)
                else:
                    t = F.interpolate(t, size=(tgt_side, tgt_side), mode='bilinear', align_corners=False)
                freqs_cis = t.squeeze(0).permute(1,2,0).flatten(0,1)
            return freqs_cis.view(1, 1, target_N, target_D)
        elif ndim == 5:
            target_H, target_W = x.shape[1], x.shape[2]
            if freqs_cis.ndim == 2:
                side = int(math.sqrt(freqs_cis.shape[0]))
                freqs_cis = freqs_cis.view(side, side, -1)
            if freqs_cis.shape[0] != target_H or freqs_cis.shape[1] != target_W:
                t = freqs_cis.permute(2,0,1).unsqueeze(0)
                if t.is_complex():
                    real = F.interpolate(t.real, size=(target_H, target_W), mode='bilinear', align_corners=False)
                    imag = F.interpolate(t.imag, size=(target_H, target_W), mode='bilinear', align_corners=False)
                    t = torch.complex(real, imag)
                else:
                    t = F.interpolate(t, size=(target_H, target_W), mode='bilinear', align_corners=False)
                freqs_cis = t.squeeze(0).permute(1,2,0)
            target_D = x.shape[4]
            return freqs_cis.view(1, target_H, target_W, 1, target_D)
        return freqs_cis
    vitdet_module.reshape_for_broadcast = fixed_reshape_for_broadcast
except ImportError: pass
# ==============================================================================

try:
    from sam3.sam.common import ResizeLongestSide
except ImportError:
    from typing import Tuple
    class ResizeLongestSide:
        def __init__(self, target_length: int) -> None:
            self.target_length = target_length
        def apply_image(self, image: np.ndarray) -> np.ndarray:
            target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
            return cv2.resize(image, (target_size[1], target_size[0]))
        def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
            old_h, old_w = original_size
            new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
            coords = coords.copy().astype(float)
            coords[..., 0] = coords[..., 0] * (new_w / old_w)
            coords[..., 1] = coords[..., 1] * (new_h / old_h)
            return coords
        def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
            boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
            return boxes.reshape(-1, 4)
        @staticmethod
        def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
            scale = long_side_length * 1.0 / max(oldh, oldw)
            newh, neww = oldh * scale, oldw * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            return (newh, neww)

class Sam3ImagePredictor:
    def __init__(self, sam_model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0) -> None:
        self.model = sam_model
        self.device = sam_model.device
        
        self.target_size = self._detect_exact_size(self.model)
        print(f"[SAM3] Initialized. Target Input Size: {self.target_size}x{self.target_size}")

        self.prompt_encoder = self._find_module_recursive(
            self.model, 
            ['geometry_encoder', 'prompt_encoder', 'sam_prompt_encoder', 'sparse_encoder'], 
            ['PromptEncoder', 'GeometryEncoder']
        )
        self.mask_decoder = self._find_module_recursive(
            self.model, 
            ['mask_decoder', 'sam_mask_decoder'], 
            ['MaskDecoder', 'SAM2MaskDecoder']
        )
        
        if not self.prompt_encoder: print("[CRITICAL] PromptEncoder NOT FOUND!")
        if not self.mask_decoder: print("[CRITICAL] MaskDecoder NOT FOUND!")

        self.transform = ResizeLongestSide(self.target_size)
        self.reset_image()

    def _find_module_recursive(self, model, valid_names, class_hints):
        for name in valid_names:
            if hasattr(model, name): return getattr(model, name)
        for name, module in model.named_children():
            if any(vn in name for vn in valid_names): return module
            if any(hint.lower() in module.__class__.__name__.lower() for hint in class_hints): return module
        for name, module in model.named_modules():
            mod_str = str(module.__class__)
            if "torch.nn" in mod_str or "TransformerDecoder" in mod_str: continue
            if any(vn in name.split('.')[-1] for vn in valid_names): return module
            if any(hint.lower() in module.__class__.__name__.lower() for hint in class_hints): return module
        return None

    def _detect_exact_size(self, model):
        for name, mod in model.named_modules():
            if hasattr(mod, 'freqs_cis') and isinstance(mod.freqs_cis, torch.Tensor):
                shape = mod.freqs_cis.shape
                if len(shape) > 0:
                    if shape[0] == 576: return 384
                    if shape[0] == 4096: return 1024
        return 384

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    @torch.no_grad()
    def set_image(self, image: np.ndarray, image_format: str = "RGB", caption: str = None) -> None:
        if image_format != "RGB": image = image[..., ::-1]
        self.reset_image()
        self.orig_h, self.orig_w = image.shape[:2]

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        self.input_h, self.input_w = input_image_torch.shape[-2:]
        h, w = self.input_h, self.input_w
        padh = max(0, self.target_size - h)
        padw = max(0, self.target_size - w)
        if padh > 0 or padw > 0:
            input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
        if input_image_torch.shape[-1] > self.target_size:
             input_image_torch = input_image_torch[..., :self.target_size, :self.target_size]

        if hasattr(self.model, "preprocess"):
            input_image_torch = self.model.preprocess(input_image_torch)
        else:
            mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).view(-1, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(-1, 1, 1)
            input_image_torch = (input_image_torch - mean) / std

        try:
            cap_input = [caption] if caption else ["face"]
            if hasattr(self.model, "backbone"):
                self.features = self.model.backbone(input_image_torch, captions=cap_input)
            elif hasattr(self.model, "forward_image"):
                self.features = self.model.forward_image(input_image_torch)
            else:
                self.features = self.model.image_encoder(input_image_torch)
        except Exception as e:
            print(f"[ERROR] Feature extraction: {e}")
            raise e
        self.is_image_set = True

    @torch.no_grad()
    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False):
        if not self.is_image_set: raise RuntimeError("Set image first.")

        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            point_coords = self.transform.apply_coords(point_coords, (self.orig_h, self.orig_w))
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, (self.orig_h, self.orig_w))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]

        if self.prompt_encoder is None: raise AttributeError("PromptEncoder not found.")

        # ======================================================================
        # [핵심] GeoPromptStub (Batch Size Enforced)
        # ======================================================================
        class GeoPromptStub:
            def __init__(self, coords, labels, boxes, masks, device):
                # 1. Determine GLOBAL Batch Size
                # boxes: [B, 4] -> 1st dim is batch
                # coords: [B, N, 2] -> 1st dim is batch
                self.bs = 1
                if boxes is not None: 
                    if boxes.ndim == 2: self.bs = boxes.shape[0]
                    elif boxes.ndim == 3: self.bs = boxes.shape[0]
                elif coords is not None: 
                    self.bs = coords.shape[0]
                
                # 2. Points Processing
                if coords is not None:
                    # [B, N, 2] -> [N, B, 2]
                    self.point_embeddings = coords.permute(1, 0, 2)
                    # [B, N] -> [B, N]
                    self.point_labels = labels
                    # [B, N]
                    self.point_mask = torch.ones(labels.shape, dtype=torch.bool, device=device)
                else:
                    # Empty Tensors MUST match batch size
                    # Embeds: [0, B, 2]
                    self.point_embeddings = torch.empty((0, self.bs, 2), device=device, dtype=torch.float)
                    # Labels: [B, 0]
                    self.point_labels = torch.empty((self.bs, 0), device=device, dtype=torch.int)
                    # Mask: [B, 0]
                    self.point_mask = torch.zeros((self.bs, 0), device=device, dtype=torch.bool)

                # 3. Boxes Processing
                if boxes is not None:
                    # Ensure [B, 1, 4] format
                    if boxes.ndim == 2: boxes = boxes.unsqueeze(1)
                    
                    # Embeds: [1, B, 4] (Seq First)
                    self.box_embeddings = boxes.permute(1, 0, 2)
                    
                    # Mask: [B, 1] (Batch First)
                    self.box_mask = torch.ones((self.bs, 1), dtype=torch.bool, device=device)
                    # Labels: [B, 1]
                    self.box_labels = torch.zeros((self.bs, 1), dtype=torch.int, device=device)
                else:
                    self.box_embeddings = torch.empty((0, self.bs, 4), device=device, dtype=torch.float)
                    self.box_mask = torch.zeros((self.bs, 0), device=device, dtype=torch.bool)
                    self.box_labels = torch.empty((self.bs, 0), device=device, dtype=torch.int)

                # 4. Masks
                self.mask_embeddings = masks
                self.mask_mask = None
                self.mask_labels = None
                if masks is not None:
                     self.mask_mask = torch.ones((self.bs, 1), dtype=torch.bool, device=device)
                     self.mask_labels = torch.zeros((self.bs, 1), dtype=torch.int, device=device)

        geo_prompt_obj = GeoPromptStub(coords_torch, labels_torch, box_torch, mask_input_torch, self.device)

        # Call Prompt Encoder
        sig = inspect.signature(self.prompt_encoder.forward)
        call_kwargs = {}
        
        feat_h, feat_w = self.input_h, self.input_w
        features_to_pass = self.features
        if isinstance(self.features, dict):
            if 'image_embed' in self.features: features_to_pass = self.features['image_embed']
            elif 'vision_features' in self.features: features_to_pass = self.features['vision_features']
            elif len(self.features) > 0: features_to_pass = list(self.features.values())[0]
        
        if isinstance(features_to_pass, torch.Tensor) and features_to_pass.ndim == 4:
            if features_to_pass.shape[1] == 256 and features_to_pass.shape[-1] != 256:
                feat_h, feat_w = features_to_pass.shape[2], features_to_pass.shape[3]
                features_to_pass = features_to_pass.permute(0, 2, 3, 1)
            else:
                feat_h, feat_w = features_to_pass.shape[1], features_to_pass.shape[2]
            
            B, H, W, C = features_to_pass.shape
            features_to_pass = features_to_pass.reshape(B, H * W, C).permute(1, 0, 2)

        if not isinstance(features_to_pass, list): call_kwargs['img_feats'] = [features_to_pass]
        else: call_kwargs['img_feats'] = features_to_pass
                
        if 'img_sizes' in sig.parameters: 
            call_kwargs['img_sizes'] = [torch.tensor([feat_h, feat_w], device=self.device)]
        if 'geo_prompt' in sig.parameters: call_kwargs['geo_prompt'] = geo_prompt_obj

        print(f"[SAM3] Calling PromptEncoder...")
        sparse_embeddings, dense_embeddings = self.prompt_encoder(**call_kwargs)
        
        # Call Mask Decoder
        if self.mask_decoder is None: raise AttributeError("MaskDecoder not found.")
        decoder_sig = inspect.signature(self.mask_decoder.forward)
        decoder_kwargs = {}

        if 'img_feats' in decoder_sig.parameters:
            decoder_kwargs['img_feats'] = call_kwargs['img_feats'] 
        elif 'image_embeddings' in decoder_sig.parameters:
            decoder_kwargs['image_embeddings'] = self.features.get("image_embed") if isinstance(self.features, dict) else self.features[0]

        if 'sparse_prompt_embeddings' in decoder_sig.parameters: decoder_kwargs['sparse_prompt_embeddings'] = sparse_embeddings
        if 'dense_prompt_embeddings' in decoder_sig.parameters: decoder_kwargs['dense_prompt_embeddings'] = dense_embeddings
        
        image_pe = None
        if hasattr(self.prompt_encoder, 'get_dense_pe'): image_pe = self.prompt_encoder.get_dense_pe()
        else: image_pe = torch.zeros((1, 256, feat_h, feat_w), device=self.device)
        if 'image_pe' in decoder_sig.parameters: decoder_kwargs['image_pe'] = image_pe

        if 'high_res_features' in decoder_sig.parameters:
             if isinstance(self.features, dict): decoder_kwargs['high_res_features'] = self.features.get("high_res_feats")
             elif isinstance(self.features, (list, tuple)) and len(self.features) > 1: decoder_kwargs['high_res_features'] = self.features[1]

        if 'multimask_output' in decoder_sig.parameters: decoder_kwargs['multimask_output'] = multimask_output
        if 'repeat_image' in decoder_sig.parameters: decoder_kwargs['repeat_image'] = False

        print(f"[SAM3] Calling MaskDecoder...")
        try:
            ret = self.mask_decoder(**decoder_kwargs)
            if isinstance(ret, tuple):
                low_res_masks = ret[0]
                iou_predictions = ret[1]
            else:
                low_res_masks = ret
                iou_predictions = None
        except Exception as e:
            print(f"[WARNING] Decoder Call Error: {e}")
            low_res_masks, iou_predictions, _ = self.mask_decoder(
                image_embeddings=self.features.get("image_embed") if isinstance(self.features, dict) else self.features[0],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

        if low_res_masks.ndim == 5: low_res_masks = low_res_masks.squeeze(1)

        masks = self.model.postprocess_masks(
            low_res_masks, self.input_h, self.input_w, self.orig_h, self.orig_w
        )

        if not return_logits:
            thresh = getattr(self.model, "mask_threshold", 0.0)
            masks = masks > thresh

        return masks[0].cpu().numpy(), iou_predictions[0].cpu().numpy(), low_res_masks[0].cpu().numpy()