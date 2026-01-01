import cv2
import numpy as np

def draw_detections(image, detections):
    """
    이미지에 탐지된 박스, 점수, 그리고 마스크(반투명)를 그립니다.
    detections: List of dicts {'box':..., 'conf':..., 'mask':...}
    """
    vis_img = image.copy()
    overlay = vis_img.copy()
    
    # 텍스트 및 선 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 색상 팔레트 (BGR)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Cyan
    ]

    # 1. 마스크 오버레이 그리기 (반투명 배경용)
    for i, det in enumerate(detections):
        mask = det.get('mask') if isinstance(det, dict) else None
        
        if mask is not None:
            color = colors[i % len(colors)]
            # 마스크 크기 안전장치
            if mask.shape[:2] != overlay.shape[:2]:
                mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 마스크 영역에 색상 적용
            overlay[mask > 0] = color

    # 2. 반투명 합성
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    # 3. 박스 및 텍스트 그리기
    for i, det in enumerate(detections):
        # 데이터 파싱
        box = None
        conf = 0.0
        
        if isinstance(det, dict):
            if 'box' in det: box = det['box']
            elif 'bbox' in det: box = det['bbox']
            conf = det.get('conf', det.get('det_score', 0.0))
        else:
            box = det[:4]
            conf = det[4] if len(det) > 4 else 0.0

        if box is None: continue

        try:
            if hasattr(conf, 'item'): conf = conf.item()
            elif isinstance(conf, (list, np.ndarray)) and len(conf) > 0: conf = conf[0]
            conf = float(conf)
        except: conf = 0.0

        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]
        
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        
        # 라벨 그리기
        label = f"#{i+1} {conf:.2f}" 
        
        # [New] LoRA 정보 표시 (확장자 제거 및 간략화)
        lora_names = det.get('lora_names', [])
        if lora_names:
            display_names = [n.replace(".safetensors", "").replace(".ckpt", "").replace(".pt", "") for n in lora_names]
            if len(display_names) > 2:
                lora_str = ",".join(display_names[:2]) + "..."
            else:
                lora_str = ",".join(display_names)
            label += f" L:{lora_str}"
        
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        y_text = y1 - 5 if y1 - 20 > 0 else y1 + h + 5
        
        cv2.rectangle(vis_img, (x1, y_text - h - 5), (x1 + w, y_text + baseline - 5), color, -1)
        
        # 텍스트 색상 (배경 밝기에 따라 자동 조정)
        text_color = (0, 0, 0) if (color[0]*0.11 + color[1]*0.59 + color[2]*0.3) > 128 else (255, 255, 255)
        cv2.putText(vis_img, label, (x1, y_text - 5), font, font_scale, text_color, 1, cv2.LINE_AA)
        
    return vis_img

def draw_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    단일 마스크를 이미지 위에 오버레이합니다. (실시간 처리 확인용)
    """
    vis = image.copy()
    overlay = vis.copy()
    
    if mask.shape[:2] != vis.shape[:2]:
        mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Mask Threshold
    if mask.max() > 1:
        mask_bool = mask > 127
    else:
        mask_bool = mask > 0.5
        
    overlay[mask_bool] = color
    
    # Blend
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    
    # Contour
    try:
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
    except Exception:
        pass
        
    return vis