import cv2
import numpy as np
import torch
import os
from core.detector import ObjectDetector

def test_detection():
    detector = ObjectDetector(model_dir=r"d:\SAM3_FaceDetailer_Ultimate\models\adetailer")
    
    # Create a dummy image (black square with a white square in the middle)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), -1)
    
    print("\n--- Testing Class Parsing (Robustness) ---")
    test_cases = [
        "person",
        "person, face",
        "person\nface",
        "  person  ,  face  \n  hand  ",
        ["person", "face"]
    ]
    
    # We'll use a standard model like yolov8n.pt which should be available or auto-download
    model_name = "yolov8n.pt"
    
    for tc in test_cases:
        print(f"\nTesting with classes input: {repr(tc)}")
        try:
            # We don't care about actual detections here as much as the logic not crashing
            # and classes being parsed correctly. 
            # In a real environment, this might download the model.
            dets = detector.detect(image, model_name, conf=0.1, classes=tc)
            print(f"Result: {len(dets)} detections found.")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    test_detection()
