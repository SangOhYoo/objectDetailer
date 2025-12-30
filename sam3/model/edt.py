import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def edt_triton(x):
    """
    Windows 호환성 패치:
    Triton 컴파일러 대신 Scipy(CPU)를 사용하여 유클리드 거리 변환(EDT)을 수행합니다.
    """
    input_device = x.device
    
    # 입력이 bool이 아니면 변환
    if x.dtype != torch.bool:
        x_bool = x > 0.5
    else:
        x_bool = x

    # CPU Numpy로 변환
    x_np = x_bool.detach().cpu().numpy()
    
    # 결과 배열 초기화
    result_np = np.zeros_like(x_np, dtype=np.float32)
    
    # 차원에 따라 처리 (Batch 처리 지원)
    # x shape이 [Batch, H, W] 또는 [H, W]라고 가정
    if x_np.ndim == 2:
        # [H, W]
        # Scipy edt는 배경(0)까지의 거리를 계산하므로 입력 반전 필요 여부 확인
        # 보통 마스크 내부 픽셀에서 가장 가까운 0까지의 거리
        result_np = distance_transform_edt(x_np)
        
    elif x_np.ndim == 3:
        # [Batch, H, W]
        for i in range(x_np.shape[0]):
            result_np[i] = distance_transform_edt(x_np[i])
            
    elif x_np.ndim == 4:
        # [Batch, Channel, H, W]
        for b in range(x_np.shape[0]):
            for c in range(x_np.shape[1]):
                result_np[b, c] = distance_transform_edt(x_np[b, c])

    # 다시 GPU 텐서로 변환
    return torch.from_numpy(result_np).to(input_device)