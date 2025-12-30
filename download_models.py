from huggingface_hub import snapshot_download
import os

# [수정] 오직 SD 1.5와 Tile 모델의 설정만 받습니다.
models_to_download = [
    "runwayml/stable-diffusion-v1-5",            # SD 1.5 표준 설정
    "lllyasviel/control_v11f1e_sd15_tile"        # 실제 사용하는 Tile 설정
]

print("=== 필수 설정 파일(Tile 전용) 다운로드 시작 ===")

for model_id in models_to_download:
    try:
        print(f"체크 중: {model_id} ...")
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            allow_patterns=["*.json", "model_index.json"], # 설정 파일만 타겟팅
            ignore_patterns=["*.safetensors", "*.bin", "*.pth"]
        )
    except Exception as e:
        print(f"!! 에러: {model_id} - {e}")

print("=== 다운로드 완료. 이제 엔진을 수정하세요. ===")