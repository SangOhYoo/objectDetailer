"""
SAM3_FaceDetailer_Ultimate Configuration Definitions
이 파일은 시스템 전역 설정과 이미지 처리 레시피(Config)를 정의합니다.
데이터 클래스(Dataclass)를 사용하여 UI와 로직 간의 데이터 전송을 엄격하게 관리합니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os

# =========================================================
# 1. 시스템 및 하드웨어 환경 설정 (System Config)
#    : 프로그램 시작 시 로드되며, 잘 변하지 않는 환경 값들
# =========================================================
@dataclass
class SystemConfig:
    """
    프로그램 전역 설정 (경로, 하드웨어 전략)
    """
    # [경로 설정]
    # 모델 파일들이 저장된 루트 경로
    model_storage_path: str = "./models"
    # 결과물이 저장될 기본 경로
    output_path: str = "./outputs"
    
    # [하드웨어 / GPU 전략]
    # "single": 1개 GPU만 사용
    # "dual_queue": 2개 GPU를 독립 큐(Queue)로 병렬 가동 (Dual RTX 3060 최적화)
    gpu_strategy: str = "dual_queue" 
    
    # [안정성 / 좀비모드]
    # True면 OOM/CUDA 에러 발생 시 해당 워커만 재부팅하고 프로그램은 유지
    auto_recover: bool = True 
    
    # [메타데이터 보존]
    # True: PIL을 사용하여 원본 EXIF 보존 + ADetailer 호환 로그 기록
    # False: OpenCV 저장 (메타데이터 삭제됨, 속도는 미세하게 빠름)
    save_metadata: bool = True
    
    # [UI 테마]
    dark_theme: bool = True


# =========================================================
# 2. 개별 이미지 처리 설정 (Detailer Config)
#    : 이미지 1장을 처리하기 위한 레시피 (ADetailer + User Custom + BMAB)
# =========================================================
@dataclass
class DetailerConfig:
    """
    이미지 처리를 위한 핵심 파라미터 집합
    UI에서 Start 버튼을 누를 때 생성되어 Queue에 담기는 객체입니다.
    """

    # -----------------------------------------------------
    # [A] 탐지 (Detection) & 타겟팅
    # -----------------------------------------------------
    # 사용할 탐지 모델 (face_yolo, hand_yolo, person_yolo, nsfw_yolo ...)
    detector_model: str = "face_yolov8n.pt"
    
    # 감지 신뢰도 (이 값 이하는 무시)
    conf_thresh: float = 0.35
    
    # 최대 탐지 개수 (사람이 100명이어도 상위 N명만 처리)
    max_det: int = 20
    
    # [스마트 필터] 이미지 크기 대비 너무 작은 얼굴(1% 미만) 무시
    min_face_ratio: float = 0.01 
    
    # [사용자 전용 기능] 성별 필터 ("All", "Male", "Female") - InsightFace 필요
    gender_filter: str = "All"
    
    # [가장자리 처리] 이미지 끝에 잘린 얼굴 무시 여부
    ignore_edge_touching: bool = False


    # -----------------------------------------------------
    # [B] 전처리 & 기하학 보정 (Preprocessing & Geometry)
    # -----------------------------------------------------
    # [ADetailer 표준] 마스크 영역 확장 (픽셀 단위)
    mask_dilation: int = 4
    
    # [ADetailer 표준] 마스크 경계 블러 (자연스러운 합성)
    mask_blur: int = 12
    
    # [ADetailer 고급] 마스크 침식 (Erosion) - 얼굴 안쪽으로 파고들기
    mask_erosion: int = 0
    
    # [★ 핵심 기술] 누운 얼굴 자동 회전 보정 (Detect -> Align 0° -> Inpaint -> Inverse)
    auto_rotate: bool = True
    
    # [★ 핵심 기술] 해부학적 검증 (InsightFace 랜드마크로 눈/코/입 순서 확인하여 괴물 필터링)
    anatomy_check: bool = True
    
    # 크롭 시 여백 (Padding) 비율 (0.25 = 상하좌우 25% 여유)
    crop_padding: float = 0.25


    # -----------------------------------------------------
    # [C] 생성 & 인페인팅 (Generation Engine)
    # -----------------------------------------------------
    # 사용할 체크포인트 파일명 (자동으로 SD1.5/SDXL 판별 로직 적용)
    checkpoint_file: str = "juggernaut_xl_v9.safetensors"
    
    # VAE 파일 (None이면 체크포인트 내장 사용)
    vae_file: Optional[str] = None
    
    # 긍정 프롬프트 (Token Chunking 자동 적용됨)
    pos_prompt: str = "best quality, detailed face, high resolution, realistic skin texture"
    
    # 부정 프롬프트
    neg_prompt: str = "(lowres, low quality, worst quality:1.2), bad anatomy, bad hands, text, watermark"
    
    # [★ 핵심 기술] YOLO 객체명/성별 정보를 프롬프트에 자동 주입 (Context Awareness)
    auto_prompt_injection: bool = True
    
    # [중요] 디노이징 강도 (0.3 ~ 0.5 권장)
    denoising_strength: float = 0.4
    
    # 샘플러 설정
    sampler_name: str = "DPM++ 2M Karras"
    steps: int = 25
    cfg_scale: float = 7.0
    seed: int = -1  # -1은 랜덤
    
    # 타겟 해상도 (1.5=512, XL=1024. 로직에서 모델에 따라 자동 조정 가능)
    target_res: int = 1024
    
    # CLIP Skip (보통 2)
    clip_skip: int = 2


    # -----------------------------------------------------
    # [D] ControlNet & 구조 제어 (BMAB Style)
    # -----------------------------------------------------
    # ControlNet 사용 여부
    use_controlnet: bool = True
    
    # 모델 선택 (tile, canny, lineart 등)
    control_model: str = "control_v11f1e_sd15_tile"
    
    # 제어 강도 (Weight)
    control_weight: float = 1.0
    
    # [BMAB 스타일] 적용 시작 시점 (Step 비율, 0.0 ~ 1.0)
    guidance_start: float = 0.0
    
    # [BMAB 스타일] 적용 종료 시점 (Step 비율, 0.0 ~ 1.0)
    # 0.4 정도로 설정하면 초반에만 뼈대를 잡고 후반엔 자유롭게 그리기 가능
    guidance_end: float = 1.0


    # -----------------------------------------------------
    # [E] 고급 & 후처리 (Advanced)
    # -----------------------------------------------------
    # [BMAB 스타일] 노이즈 주입 (피부 질감 향상, Noise Alpha)
    noise_multiplier: float = 1.0
    
    # 색감 보정 (인페인팅 후 색 틀어짐 방지) - "None", "Wavelet", "Adain"
    color_fix: str = "None"
    
    # Hires Fix (전처리 업스케일) 사용 여부
    use_hires_fix: bool = False
    
    # Upscale 배율 (1.5 ~ 2.0)
    upscale_factor: float = 1.5
    
    # -----------------------------------------------------
    # [Helper Method] ADetailer 호환용 딕셔너리 변환
    # -----------------------------------------------------
    def to_adetailer_json(self) -> Dict[str, Any]:
        """
        이 설정을 ADetailer가 인식할 수 있는 JSON 구조로 변환합니다.
        메타데이터 저장 시(Exif/PNG Info) 사용됩니다.
        """
        return {
            "ad_model": self.detector_model,
            "ad_prompt": self.pos_prompt,
            "ad_negative_prompt": self.neg_prompt,
            "ad_confidence": self.conf_thresh,
            "ad_mask_blur": self.mask_blur,
            "ad_denoising_strength": self.denoising_strength,
            "ad_inpaint_only_masked": True,
            "ad_inpaint_only_masked_padding": self.mask_dilation,
            "ad_use_inpaint_width_height": True,
            "ad_inpaint_width": self.target_res,
            "ad_inpaint_height": self.target_res,
            "ad_use_steps": True,
            "ad_steps": self.steps,
            "ad_use_cfg_scale": True,
            "ad_cfg_scale": self.cfg_scale,
            "ad_controlnet_model": self.control_model if self.use_controlnet else "None",
            # SAM3 고유 기능도 기록 (추후 분석용)
            "sam3_auto_rotate": self.auto_rotate,
            "sam3_anatomy_check": self.anatomy_check,
            "sam3_guidance_end": self.guidance_end
        }