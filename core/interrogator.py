import os
import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

class Interrogator:
    def __init__(self, model_path=None, tags_path=None, device="cpu"):
        self.device = device
        # Default paths (Download if missing)
        base_dir = r"D:\AI_Models\Interrogators\WD14"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            
        self.model_path = model_path or os.path.join(base_dir, "model.onnx")
        self.tags_path = tags_path or os.path.join(base_dir, "selected_tags.csv")
        
        self.session = None
        self.tags_df = None
        
    def _download(self, url, path):
        print(f"[Interrogator] Downloading {os.path.basename(path)} from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(path)) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

    def load_model(self):
        if self.session is not None:
            return
            
        # Download if missing (HuggingFace Mirrors or Direct)
        if not os.path.exists(self.model_path):
            url = "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/model.onnx"
            self._download(url, self.model_path)
            
        if not os.path.exists(self.tags_path):
            url = "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/selected_tags.csv"
            self._download(url, self.tags_path)
            
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if self.device == "cuda":
            providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
            
        print(f"[Interrogator] Loading WD14 Tagger from {self.model_path}...")
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.tags_df = pd.read_csv(self.tags_path)
        
    def preprocess(self, image: np.ndarray):
        # Image is BGR, WD14 Vit V2 expects RGB 448x448 (Vit) or 224x224 (older)
        # Checking input shape of ONNX
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape # [1, H, W, 3] or [1, 3, H, W]?
        # WD14 models usually use [1, 448, 448, 3] NHWC
        target_h, target_w = input_shape[1], input_shape[2]
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Letterbox/Resize
        h, w = img.shape[:2]
        m = max(h, w)
        pad_h = (m - h) // 2
        pad_w = (m - w) // 2
        img = cv2.copyMakeBorder(img, pad_h, m - h - pad_h, pad_w, m - w - pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        img = img.astype(np.float32)
        # Model expects raw 0-255 float usually, or normalize?
        # SmilingWolf models expect [0, 255] float
        img = np.expand_dims(img, axis=0)
        return img

    def interrogate(self, image: np.ndarray, threshold: float = 0.35) -> str:
        try:
            self.load_model()
            blob = self.preprocess(image)
            
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            probs = self.session.run([output_name], {input_name: blob})[0][0]
            
            # Index 0-3 are ratings (General, Sensitive, Questionable, Explicit) - skip
            # General tags usually start from index 4
            results = []
            for i, p in enumerate(probs):
                if p >= threshold:
                    row = self.tags_df.iloc[i]
                    tag = row['name']
                    category = row['category'] # 0 for general tags
                    
                    if category == 0:
                        results.append(tag)
            
            # Clean up tags (replace underscores, etc)
            clean_tags = [t.replace('_', ' ') for t in results]
            
            # Filter out some very generic or unwanted ones
            # (Optional: user can provide blacklist)
            
            prompt = ", ".join(clean_tags)
            return prompt
            
        except Exception as e:
            print(f"[Interrogator] Error: {e}")
            return ""
