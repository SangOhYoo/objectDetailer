import os
import cv2
import numpy as np
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper
from core.metadata import save_image_with_metadata

class MockConfig:
    def to_adetailer_json(self):
        return {
            "pos_prompt": "face detail",
            "neg_prompt": "blurry face",
            "model": "face_yolov8n.pt",
            "conf": 0.5
        }

def create_test_image(path, params_text, has_exif=True):
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters", params_text)
    
    exif_bytes = b""
    if has_exif:
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
        exif_dict["0th"][piexif.ImageIFD.Make] = "Original Camera"
        user_comment = piexif.helper.UserComment.dump(params_text, encoding="unicode")
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
        exif_bytes = piexif.dump(exif_dict)
    
    img.save(path, pnginfo=metadata, exif=exif_bytes)
    print(f"Created test image: {path} with params: {params_text[:50]}...")

def verify_metadata():
    input_path = "test_meta_input.png"
    output_path = "test_meta_output.png"
    
    original_params = "Original Positive\nNegative prompt: Original Negative\nSteps: 20, Sampler: Euler a"
    create_test_image(input_path, original_params)
    
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    print("Running save_image_with_metadata...")
    save_image_with_metadata(test_img, input_path, output_path)

    
    # Check output
    with Image.open(output_path) as out_img:
        out_params = out_img.info.get("parameters", "")
        print(f"Output parameters:\n{out_params}")
        
        # Check if original is present
        if "Original Positive" in out_params:
            print("[SUCCESS] Original params preserved in PNG!")
        else:
            print("[FAIL] Original params LOST in PNG!")

        # Check EXIF
        raw_exif = out_img.info.get("exif")
        if raw_exif:
            exif_dict = piexif.load(raw_exif)
            user_comment_raw = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            if user_comment_raw:
                user_comment = piexif.helper.UserComment.load(user_comment_raw)
                print(f"Output UserComment:\n{user_comment}")
                if "Original Positive" in user_comment:
                    print("[SUCCESS] Original params preserved in EXIF!")
                else:
                    print("[FAIL] Original params LOST in EXIF!")
            else:
                print("[FAIL] UserComment missing in output EXIF!")
        else:
            print("[FAIL] EXIF missing in output!")

    # Cleanup
    try:
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)
    except: pass

if __name__ == "__main__":
    verify_metadata()
