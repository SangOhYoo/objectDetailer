import cv2
import numpy as np
import math

# OpenPose COCO format pairs and colors
# (point_idx_1, point_idx_2, color_tuple)
# Colors are typically (B, G, R) in OpenCV
POSE_PAIRS = [
    # Nose, Eyes, Ears
    (0, 1, (255, 0, 0)), (0, 2, (0, 0, 255)), (1, 3, (255, 0, 0)), (2, 4, (0, 0, 255)),
    # Shoulders
    (5, 6, (255, 0, 255)), 
    # Arms
    (5, 7, (255, 0, 0)), (7, 9, (255, 0, 0)), 
    (6, 8, (0, 0, 255)), (8, 10, (0, 0, 255)),
    # Body
    (11, 12, (0, 255, 0)), (5, 11, (255, 0, 255)), (6, 12, (255, 0, 255)),
    # Legs
    (11, 13, (255, 0, 0)), (13, 15, (255, 0, 0)),
    (12, 14, (0, 0, 255)), (14, 16, (0, 0, 255))
]

# Keypoint Colors (17 points)
# 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
# 5: L-Sho, 6: R-Sho, 7: L-Elb, 8: R-Elb, 9: L-Wri, 10: R-Wri
# 11: L-Hip, 12: R-Hip, 13: L-Knee, 14: R-Knee, 15: L-Ank, 16: R-Ank
KP_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), # 0-4
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), # 5-9
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), # 10-14
    (255, 0, 255), (255, 0, 170)                                           # 15-16
]

def draw_pose_map(keypoints, height, width):
    """
    Draws an OpenPose-style skeleton on a black background.
    
    Args:
        keypoints (np.ndarray or list): Shape (17, 3) or (17, 2). [x, y, conf]
        height (int): Canvas height
        width (int): Canvas width
        
    Returns:
        np.ndarray: BGR image of shape (height, width, 3)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process each person (if list of persons provided) or single person
    # Here we assume keypoints is a single person's (17, 3) array mainly,
    # but if multiple, iterating is handled by caller or we can handle list of arrays here.
    # Let's assume input is a list of persons, where each person is (17, 3) array.
    
    persons = []
    if isinstance(keypoints, list):
        if len(keypoints) > 0:
             # Check if the first element is a Person (len 17) or a Point (len 2 or 3)
             first = keypoints[0]
             if isinstance(first, (list, np.ndarray)) and len(first) == 17:
                 # It's a list of Persons
                 persons = keypoints
             else:
                 # It's a single Person (list of points)
                 persons = [keypoints]
    elif isinstance(keypoints, np.ndarray):
        if keypoints.ndim == 2 and keypoints.shape[0] == 17:
            persons = [keypoints]
        elif keypoints.ndim == 3:
            persons = keypoints
            
    for kp in persons:
        kp = np.array(kp)
        
        # Draw Limbs
        for i, (p1, p2, color) in enumerate(POSE_PAIRS):
            if p1 >= len(kp) or p2 >= len(kp): continue
            
            x1, y1 = int(kp[p1][0]), int(kp[p1][1])
            x2, y2 = int(kp[p2][0]), int(kp[p2][1])
            c1, c2 = kp[p1][2], kp[p2][2] # Confidence
            
            if c1 > 0.3 and c2 > 0.3: # Threshold
                # BGR to RGB? ControlNet usually expects OpenPose typical colors.
                # The above colors are standard OpenPose BGR.
                cv2.line(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                
        # Draw Keypoints
        for i, point in enumerate(kp):
            x, y = int(point[0]), int(point[1])
            c = point[2]
            if c > 0.3:
                cv2.circle(canvas, (x, y), 4, KP_COLORS[i], -1, cv2.LINE_AA)
                
    return canvas

def create_pose_map_from_detections(detections, height, width):
    """
    Wrapper to handle multiple detections from detector.detect_pose
    """
    all_kps = [d['keypoints'] for d in detections if 'keypoints' in d]
    return draw_pose_map(all_kps, height, width)
