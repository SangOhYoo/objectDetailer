# ObjectDetailer (Standalone)

**ObjectDetailer** is a standalone desktop application designed to automatically detect, mask, and refinement specific objects (such as faces, hands, or people) within images using Generative AI. 
It implements the functionality of the popular "ADetailer" extension as an independent tool, powered by Stable Diffusion, YOLO, and SAM (Segment Anything Model).

**ObjectDetailer**는 이미지 내의 특정 객체(얼굴, 손, 사람 등)를 자동으로 탐지하고 마스킹하여 생성형 AI로 디테일을 보정해주는 독립형 데스크톱 애플리케이션입니다.
Stable Diffusion, YOLO, SAM(Segment Anything Model) 기술을 기반으로 하며, 기존 ADetailer 확장 기능을 별도의 웹 UI 없이 로컬에서 독립적으로 실행할 수 있도록 구현했습니다.

---

## ✨ Key Features (주요 기능)

* **Auto-Detection**: Automatically detects objects using state-of-the-art models (YOLO, MediaPipe, etc.).
    * **자동 탐지**: 최신 객체 탐지 모델을 사용하여 이미지 내 객체를 자동으로 찾아냅니다.
* **Smart Segmentation**: Utilizes the Segment Anything Model (SAM) for pixel-perfect masking.
    * **정교한 세그멘테이션**: SAM을 활용하여 탐지된 객체의 외곽선을 정밀하게 따냅니다.
* **AI Inpainting**: Refines and regenerates detected areas using Stable Diffusion for higher quality details.
    * **AI 인페인팅**: Stable Diffusion을 사용하여 탐지된 영역을 고화질로 다시 그려 디테일을 향상시킵니다.
* **Standalone GUI**: User-friendly interface built with PySide6/PyQt, requiring no browser or WebUI.
    * **독립형 GUI**: 웹 브라우저나 복잡한 WebUI 설치 없이 직관적인 데스크톱 인터페이스를 제공합니다.
* **Multi-Model Support**: Supports various checkpoints and LoRAs for customized generation.
    * **다양한 모델 지원**: 사용자가 원하는 체크포인트와 LoRA를 적용하여 스타일을 커스텀할 수 있습니다.

---

## 🛠️ Prerequisites (준비 사항)

* **OS**: Windows 10/11 (Recommended), Linux
* **Python**: 3.10+
* **GPU**: NVIDIA GPU with CUDA support (Minimum 8GB VRAM recommended)
    * NVIDIA GPU 및 CUDA 환경 (최소 8GB VRAM 권장)

---

## 🚀 Installation (설치 방법)

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/ObjectDetailer.git](https://github.com/your-username/ObjectDetailer.git)
    cd ObjectDetailer
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have the correct version of PyTorch installed for your CUDA version.)*

4.  **Download Models**
    Run the included script to download necessary model weights (YOLO, SAM, etc.).
    ```bash
    python download_models.py
    ```

---

## 💻 Usage (사용 방법)

### Quick Start (Windows)
Simply run the `run.bat` file.
`run.bat` 파일을 실행하면 가상환경 진입부터 실행까지 자동으로 진행됩니다.

### Manual Start
```bash
python main.py# objectDetailer

### Workflow

1. **Load Image**: Drag and drop an image or use the "Open" button.
2. **Configuration**:
* Select the **Detection Model** (e.g., face_yolo, person_yolo).
* Input your **Prompt** (e.g., "highly detailed face, beautiful eyes").
* Adjust **Denoising Strength** and **Inpaint Settings**.


3. **Process**: Click the **"Run"** button.
4. **Save**: The processed image will be displayed and can be saved to your local drive.

---

## 📂 Project Structure (프로젝트 구조)

```
ObjectDetailer/
├── core/               # Core logic (Backend)
│   ├── detector.py     # Object detection logic (YOLO/MediaPipe)
│   ├── sam_wrapper.py  # Segment Anything Model wrapper
│   ├── sd_engine.py    # Stable Diffusion inference engine
│   └── pipeline.py     # Main processing pipeline
├── ui/                 # User Interface (Frontend)
│   ├── main_window.py  # Main GUI implementation
│   └── components.py   # UI widgets
├── configs/            # Configuration files (Model presets, Datasets)
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── download_models.py  # Model downloader script

```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
버그 제보나 기능 개선 요청은 언제나 환영합니다. Pull Request를 통해 기여해 주세요.

## 📄 License

This project is licensed under the MIT License.
이 프로젝트는 MIT 라이선스를 따릅니다.

```

---

### 💡 추가 제안 사항 (Next Steps)

1.  **`requirements.txt` 버전 고정**: 다른 사용자가 설치할 때 충돌이 없도록 `torch`, `diffusers`, `ultralytics` 등의 라이브러리 버전을 명시하는 것이 좋습니다.
2.  **스크린샷 추가**: `2026-01-01 08 39 00.png` 파일을 `assets` 폴더 등을 만들어 넣고, README 상단에 데모 이미지로 삽입하면 프로젝트 이해도가 훨씬 높아집니다.
    * 예: `![Demo Image](assets/2026-01-01 08 39 00.png)`
3.  **Config 문서화**: `configs/` 폴더 안의 YAML 파일들을 사용자가 어떻게 수정하여 커스텀할 수 있는지에 대한 가이드(Wiki 등)가 있으면 더 좋습니다.

이 문서를 바로 `README.md`에 복사해서 사용하시면 됩니다! 더 수정이 필요한 부분이 있다면 알려주세요.

```
