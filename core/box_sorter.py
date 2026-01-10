import numpy as np

def sort_boxes(boxes, scores, method, image_width, image_height):
    """
    바운딩 박스를 지정된 기준에 따라 정렬합니다.

    Args:
        boxes (list or np.ndarray): [x1, y1, x2, y2] 형식의 바운딩 박스 목록
        scores (list or np.ndarray): 각 박스의 신뢰도 점수
        method (str): 정렬 기준 ('위치(좌에서 우)', '위치 (중앙에서 바깥)', '영역 (대형에서 소형)', '신뢰도')
        image_width (int): 이미지 너비
        image_height (int): 이미지 높이

    Returns:
        tuple: (정렬된 박스, 정렬된 점수, 정렬된 인덱스)
    """
    if boxes is None or len(boxes) == 0:
        return boxes, scores, []

    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = np.arange(len(boxes))

    if method == "위치(좌에서 우)":
        # x1 좌표 기준 오름차순 정렬
        sort_idx = np.argsort(boxes[:, 0])

    elif method == "위치(우에서 좌)": # [New]
        # x1 좌표 기준 내림차순 정렬
        sort_idx = np.argsort(boxes[:, 0])[::-1]

    elif method == "위치 (중앙에서 바깥)":
        # 박스 중심점 계산
        box_cx = (boxes[:, 0] + boxes[:, 2]) / 2
        box_cy = (boxes[:, 1] + boxes[:, 3]) / 2
        
        # 이미지 중심점
        img_cx = image_width / 2
        img_cy = image_height / 2
        
        # 이미지 중심과의 유클리드 거리 계산 (오름차순)
        distances = (box_cx - img_cx)**2 + (box_cy - img_cy)**2
        sort_idx = np.argsort(distances)

    elif method == "위치(위에서 아래)": # [New]
        # y1 좌표 기준 오름차순 정렬
        sort_idx = np.argsort(boxes[:, 1])

    elif method == "위치(아래에서 위)": # [New]
        # y1 좌표 기준 내림차순 정렬
        sort_idx = np.argsort(boxes[:, 1])[::-1]

    elif method == "영역 (대형에서 소형)":
        # 면적 계산: (x2 - x1) * (y2 - y1)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # 내림차순 정렬
        sort_idx = np.argsort(areas)[::-1]

    else: # "신뢰도" 또는 기본값
        # 점수 기준 내림차순 정렬
        sort_idx = np.argsort(scores)[::-1]

    return boxes[sort_idx], scores[sort_idx], sort_idx