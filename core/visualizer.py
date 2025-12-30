import cv2
import numpy as np

def draw_detections(image, detections):
    """
    이미지에 탐지된 박스와 점수를 그립니다. (노란색 박스용)
    detections: [(x1, y1, x2, y2, conf), ...] 형태의 리스트
    """
    vis_img = image.copy()
    
    # 텍스트 및 선 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    box_color = (0, 255, 255) # 노란색 (BGR: Yellow)
    text_color = (0, 0, 0)    # 검은색 텍스트
    text_bg_color = (0, 255, 255) # 텍스트 배경 (노란색)
    
    for det in detections:
        # 1. 좌표 및 점수 추출
        if isinstance(det, dict): # InsightFace 포맷
            box = list(map(int, det['bbox']))
            conf = det.get('det_score', 0.0)
        else: # YOLO 포맷 (List/Tuple/ndarray)
            box = list(map(int, det[:4]))
            # 인덱스 4가 있는지 확인 후 가져옴
            conf = det[4] if len(det) > 4 else 0.0

        # [CRITICAL FIX] 중첩된 배열(예: [[0.8]])까지 완벽하게 스칼라로 변환
        try:
            # NumPy 배열이거나 텐서인 경우
            if hasattr(conf, 'item'):
                conf = conf.item()
            # 그래도 배열이라면(크기가 1이 아닌 경우 등), 첫 번째 원소 강제 추출
            elif isinstance(conf, (list, np.ndarray)):
                while isinstance(conf, (list, np.ndarray)) and len(conf) > 0:
                    conf = conf[0]
            
            conf = float(conf)
        except Exception:
            conf = 0.0 # 변환 실패 시 기본값

        x1, y1, x2, y2 = box
        
        # 2. 박스 그리기
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, thickness)
        
        # 3. 라벨(점수) 그리기
        label = f"{conf:.2f}" 
        
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        y_text = y1 - 5 if y1 - 20 > 0 else y1 + h + 5
        
        cv2.rectangle(vis_img, (x1, y_text - h - 5), (x1 + w, y_text + baseline - 5), text_bg_color, -1)
        cv2.putText(vis_img, label, (x1, y_text - 5), font, font_scale, text_color, 1, cv2.LINE_AA)
        
    return vis_img