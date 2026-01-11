"""
2: ObjectDetailer_Ultimate Configuration Schemas
이 파일은 시스템 설정과 디테일러 설정의 데이터 구조(Dataclass)를 정의합니다.
현재 시스템은 유연성을 위해 Dictionary 기반의 설정을 주로 사용하지만,
이 클래스들은 설정 값의 타입과 기본값을 참조하는 스키마(Reference Schema)로 활용됩니다.
UI에서 사용되는 모든 키(Key)를 포함하도록 최신화되었습니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os

# =========================================================
# 1. 시스템 및 하드웨어 환경 설정 (System Config)
# =========================================================
@dataclass
class SystemConfig:
    """
    프로그램 전역 설정 (경로, 하드웨어 전략)
    """
    model_storage_path: str = "./models"
    output_path: str = "./outputs"
    gpu_strategy: str = "dual_queue" 
    auto_recover: bool = True 
    save_metadata: bool = True
    dark_theme: bool = True


# =========================================================
# 2. 개별 이미지 처리 설정 (Detailer Config)
# =========================================================
@dataclass
class DetailerConfig:
    """
    이미지 처리를 위한 핵심 파라미터 집합
    UI의 모든 설정값을 포함합니다.
    """

    # -----------------------------------------------------
    # [A] 탐지 (Detection) & 타겟팅
    # -----------------------------------------------------
    detector_model: str = "face_yolov8n.pt"
    yolo_classes: str = "" # [New] UI Text Input
    conf_thresh: float = 0.35
    max_det: int = 20
    min_face_ratio: float = 1.0 # % unit in UI logic
    max_face_ratio: float = 100.0
    gender_filter: str = "All"
    ignore_edge_touching: bool = False
    
    # SAM (Segment Anything)
    use_sam: bool = False
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.95
    
    # Pose / Rotation
    use_pose_rotation: bool = False # [New] Pose Rotation

    # -----------------------------------------------------
    # [B] 전처리 & 마스크 (Preprocessing & Mask)
    # -----------------------------------------------------
    mask_dilation: int = 4
    mask_blur: int = 12
    mask_erosion: int = 0
    x_offset: int = 0
    y_offset: int = 0
    
    # [New] Mask Content & Area
    mask_content: str = "original"
    inpaint_full_res: bool = False
    
    auto_rotate: bool = True
    anatomy_check: bool = True
    crop_padding: float = 32 # Pixel unit in UI

    # -----------------------------------------------------
    # [C] 생성 & 인페인팅 (Generation Engine)
    # -----------------------------------------------------
    # Global Defaults (used if Separate Pass is OFF)
    checkpoint_file: str = "juggernaut_xl_v9.safetensors" # Maps to UI Combo if not separate
    vae_file: Optional[str] = None
    
    pos_prompt: str = ""
    neg_prompt: str = ""
    auto_prompt_injection: bool = True
    
    denoising_strength: float = 0.4
    context_expand_factor: float = 1.0
    
    sampler_name: str = "DPM++ 2M Karras"
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1 
    clip_skip: int = 2
    
    # Resolution (0 means auto/default)
    inpaint_width: int = 0
    inpaint_height: int = 0
    
    # Advanced / Effect
    use_hires_fix: bool = False
    
    # [New] Hires Fix (Reforge Style)
    hires_upscaler: str = "None"
    hires_upscale_factor: float = 1.5
    hires_steps: int = 14
    hires_denoise: float = 0.4
    hires_cfg: float = 5.0
    hires_width: int = 0 # 0 means disabled/auto
    hires_height: int = 0
    
    restore_face: bool = False
    use_noise_mask: bool = False
    mask_merge_mode: str = "None" # "None", "Merge", "Merge and Invert"
    color_fix: str = "None"

    # [New] Soft Inpainting
    use_soft_inpainting: bool = False
    soft_schedule_bias: float = 1.0
    soft_preservation_strength: float = 0.5
    soft_transition_contrast: float = 4.0
    soft_mask_influence: float = 0.0
    soft_diff_threshold: float = 0.5
    soft_diff_contrast: float = 2.0

    # -----------------------------------------------------
    # [D] ControlNet
    # -----------------------------------------------------
    use_controlnet: bool = True # Implied by model selection != None
    control_model: str = "None"
    control_module: str = "None" # Preprocessor
    control_weight: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0

    # -----------------------------------------------------
    # [E] BMAB (Effects)
    # -----------------------------------------------------
    bmab_enabled: bool = True
    bmab_contrast: float = 1.0
    bmab_brightness: float = 1.0
    bmab_sharpness: float = 1.0
    bmab_color_temp: float = 0.0
    bmab_noise_alpha: float = 0.0
    bmab_edge_strength: float = 0.0
    bmab_edge_low: int = 50
    bmab_edge_high: int = 200
    bmab_edge_enabled: bool = False # [New]
    
    # -----------------------------------------------------
    # [F] Composition (Resize)
    # -----------------------------------------------------
    resize_enable: bool = False
    resize_ratio: float = 0.6
    resize_align: str = "Center"
    bmab_landscape_detail: bool = False # [New]

    # -----------------------------------------------------
    # [G] Detail Daemon (DD)
    # -----------------------------------------------------
    dd_enabled: bool = False
    dd_amount: float = 0.1
    dd_start: float = 0.2
    dd_end: float = 0.8
    dd_start_offset: float = 0.0
    dd_end_offset: float = 0.0
    dd_bias: float = 0.5
    dd_exponent: float = 1.0
    dd_fade: float = 0.0
    dd_smooth: bool = True
    dd_mode: str = "both"
    dd_hires: bool = False

    # -----------------------------------------------------
    # [H] 고급 분리 설정 (Separate Pass Overrides)
    # -----------------------------------------------------
    sep_ckpt: bool = False
    sep_ckpt_name: str = "Use Global"
    sep_vae: bool = False
    sep_vae_name: str = "Use Global"
    
    sep_sampler: bool = False
    # sampler_name is reused or separate logic handles it
    
    sep_steps: bool = False
    # steps reused locally if checked
    
    sep_cfg: bool = False
    # cfg_scale reused locally if checked
    
    sep_clip: bool = False
    # clip_skip reused locally if checked
    
    sep_noise: bool = False # Separate Noise Generation

