import cv2
import numpy as np

def apply_color_fix(target_img, source_img, method="None"):
    """
    인페인팅 결과(target)의 색감을 원본(source)에 맞게 보정합니다.
    """
    if method == "Wavelet":
        return wavelet_color_fix(target_img, source_img)
    elif method == "Adain":
        return adain_color_fix(target_img, source_img)
    elif method == "Histogram":
        return histogram_matching(target_img, source_img)
    elif method == "Linear":
        return rgb_linear_fix(target_img, source_img)
    return target_img

def histogram_matching(target, source):
    """CDF(Cumulative Distribution Function)를 기반으로 히스토그램을 일치시킵니다."""
    result = np.zeros_like(target)
    for i in range(3): # B, G, R channels
        # Source CDF
        s_values, s_counts = np.unique(source[..., i], return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        
        # Target CDF
        t_values, t_counts = np.unique(target[..., i], return_counts=True)
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        
        # Mapping
        interp_t_values = np.interp(t_quantiles, s_quantiles, s_values)
        lookup_table = np.interp(np.arange(256), t_values, interp_t_values).astype(np.uint8)
        result[..., i] = cv2.LUT(target[..., i], lookup_table)
    return result

def rgb_linear_fix(target, source):
    """RGB 각 채널의 평균과 표준편차를 매칭 (Linear Alignment)"""
    target = target.astype(np.float32)
    source = source.astype(np.float32)
    
    result = np.zeros_like(target)
    for i in range(3):
        s_mean, s_std = cv2.meanStdDev(source[..., i])
        t_mean, t_std = cv2.meanStdDev(target[..., i])
        
        s_mean, s_std = s_mean.item(), s_std.item()
        t_mean, t_std = t_mean.item(), t_std.item()
        
        if t_std < 1e-5: t_std = 1e-5
        
        # (x - t_mean) * (s_std / t_std) + s_mean
        result[..., i] = (target[..., i] - t_mean) * (s_std / t_std) + s_mean
        
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def adain_color_fix(target, source):
    """Adaptive Instance Normalization: 평균과 표준편차를 매칭"""
    # LAB 색상 공간으로 변환 (밝기와 색상 분리)
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    source_mean = source_mean.flatten()
    source_std = source_std.flatten()
    target_mean = target_mean.flatten()
    target_std = target_std.flatten()
    
    # 0으로 나누기 방지
    target_std[target_std == 0] = 1e-5

    # 채널별 통계 매칭
    result_lab = np.zeros_like(target_lab)
    for i in range(3):
        result_lab[..., i] = (target_lab[..., i] - target_mean[i]) * (source_std[i] / target_std[i]) + source_mean[i]

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def wavelet_color_fix(target, source):
    """Wavelet (Frequency Separation): 원본의 저주파(색감) + 결과의 고주파(디테일) 합성"""
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 가우시안 블러를 이용한 저주파 추출
    source_low = cv2.GaussianBlur(source_lab, (0, 0), 5)
    target_low = cv2.GaussianBlur(target_lab, (0, 0), 5)
    
    target_high = target_lab - target_low
    result_lab = source_low + target_high
    
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)