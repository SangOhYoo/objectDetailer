import os
import numpy as np
import cv2

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sam_model_registry = None
    SamPredictor = None
    print("[SAM] Error: 'segment_anything' library not found.")

class SamInference:
    def __init__(self, model_type="vit_b", checkpoint=None, device="cuda"):
        """
        :param checkpoint: config.yaml에서 전달받은 모델 절대 경로
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint = checkpoint
        
        self.predictor = None
        self.is_loaded = False
        self.is_image_set = False

        if sam_model_registry is not None:
            self.load_model()

    def load_model(self):
        if self.is_loaded: return

        if not self.checkpoint or not os.path.exists(self.checkpoint):
            print(f"[SAM] Error: Checkpoint not found at {self.checkpoint}")
            return

        print(f"[SAM] Loading {self.model_type} from {self.checkpoint}...")
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self.is_loaded = True
            print(f"[SAM] Loaded on {self.device}")
        except Exception as e:
            print(f"[SAM] Load failed: {e}")

    def set_image(self, image_bgr: np.ndarray):
        if not self.is_loaded or self.predictor is None:
            self.load_model()
            if not self.is_loaded: return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        self.is_image_set = True

    def predict_mask_from_box(self, box: list):
        if not self.is_image_set:
            print("[SAM] Error: set_image() must be called first.")
            return None

        input_box = np.array(box)
        masks, _, _ = self.predictor.predict(
            point_coords=None, point_labels=None,
            box=input_box[None, :], multimask_output=False
        )
        
        return (masks[0] * 255).astype(np.uint8)