@echo off
setlocal enabledelayedexpansion

:: ------------------------------------------------------------------
:: [설정] 프로젝트 변수 정의
:: ------------------------------------------------------------------
set "PROJECT_NAME=SAM3_FaceDetailer_Ultimate"
set "VENV_DIR=venv"
set "PYTHON_EXEC=python"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
:: [Dual GPU 최적화] 각 워커가 CPU 코어를 독점하지 않도록 제한 (병목 방지)
set "OMP_NUM_THREADS=1"

:: 콘솔 한글 깨짐 방지 (UTF-8)
chcp 65001 > nul

echo ================================================================
echo  🚀 %PROJECT_NAME% Launcher
echo ================================================================

:: 1. 가상환경(venv) 확인 및 생성
if not exist "%VENV_DIR%" (
    echo [INFO] 가상환경 폴더 '%VENV_DIR%' 가 없습니다. 새로 생성합니다...
    %PYTHON_EXEC% -m venv %VENV_DIR%
    
    if errorlevel 1 (
        echo [ERROR] 가상환경 생성 실패 - Python 3.10 이상이 설치되어 있는지 확인하세요.
        pause
        exit /b
    )
    echo [INFO] 가상환경 생성 완료.
)

:: 2. 가상환경 활성화
call %VENV_DIR%\Scripts\activate.bat

:: pip 및 setuptools 업데이트 (설치 안정성 확보)
%PYTHON_EXEC% -m pip install --upgrade pip setuptools

:: PyTorch (CUDA 12.1) 설치 확인
%PYTHON_EXEC% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyTorch CUDA 12.1 환경을 설치/복구합니다...
    %PYTHON_EXEC% -m pip install "torch>=2.1.0" "torchvision>=0.16.0" --index-url https://download.pytorch.org/whl/cu121
)

:: 3. 필수 패키지 설치 및 업데이트
if exist "requirements.txt" (
    echo [INFO] 라이브러리 의존성을 확인합니다...
    :: -q 옵션으로 이미 설치된 항목은 조용히 넘어감
    pip install -r requirements.txt
    
    if errorlevel 1 (
        echo [WARNING] 라이브러리 설치 중 오류가 발생했습니다. - 네트워크 상태 확인 필요
        pause
    )

    :: MediaPipe 초기화 오류 방지를 위한 1회 강제 재설치
    if not exist "%VENV_DIR%\.mediapipe_fixed_v7" (
        echo [INFO] Windows 환경 MediaPipe 호환성 패치를 적용합니다... - v7
        :: 충돌 방지를 위해 관련 패키지 제거 후 재설치 (NumPy 버전 고정 포함)
        pip uninstall -y mediapipe protobuf numpy
        :: 안정적인 버전(0.10.14) 및 런타임 라이브러리 추가 설치
        pip install --no-cache-dir --force-reinstall "mediapipe==0.10.14" "protobuf<5" "numpy<2" "msvc-runtime"
        echo fixed > "%VENV_DIR%\.mediapipe_fixed_v7"
    )
) else (
    echo [WARNING] 'requirements.txt' 파일이 없습니다. 패키지 설치를 건너뜁니다.
)

:: 4. 메인 프로그램 실행
echo.
echo [INFO] 메인 프로그램 main.py 을 실행합니다...
echo ----------------------------------------------------------------

python main.py

:: 5. 종료 후 처리
if errorlevel 1 (
    echo.
    echo ----------------------------------------------------------------
    echo [ERROR] 프로그램이 비정상적으로 종료되었습니다.
    echo 위 오류 메시지를 확인해 주세요.
    pause
) else (
    echo.
    echo [INFO] 프로그램이 정상 종료되었습니다.
    timeout /t 3 > nul
)