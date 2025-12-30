@echo off
setlocal enabledelayedexpansion

:: ------------------------------------------------------------------
:: [설정] 프로젝트 이름 및 환경 변수
:: ------------------------------------------------------------------
set "PROJECT_NAME=SAM3_FaceDetailer_Ultimate"
set "VENV_DIR=venv"
set "PYTHON_EXEC=python"

:: 콘솔 한글 깨짐 방지 (UTF-8)
chcp 65001 > nul

echo ================================================================
echo  %PROJECT_NAME% Launcher
echo ================================================================

:: 1. 가상환경 존재 여부 확인 및 생성
if not exist "%VENV_DIR%" (
    echo [INFO] 가상환경 venv 폴더가 없습니다. 새로 생성합니다...
    %PYTHON_EXEC% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] 가상환경 생성 실패. Python 설치를 확인하세요.
        pause
        exit /b
    )
    echo [INFO] 가상환경 생성 완료.
)

:: 2. 가상환경 활성화
call %VENV_DIR%\Scripts\activate.bat

:: 3. 필수 패키지 설치
if exist "requirements.txt" (
    echo [INFO] 패키지 의존성을 확인합니다...
    :: 이미 설치된 패키지 메시지는 숨기고 설치/업데이트 내용만 출력
    pip install -r requirements.txt | findstr /V "Requirement already satisfied"
) else (
    echo [WARNING] requirements.txt 파일이 없습니다. 패키지 설치를 건너뜁니다.
)

:: 4. 실행 전 시스템 체크
echo.
echo [SYSTEM CHECK]
python -c "import torch; print(f' - PyTorch: {torch.__version__}'); print(f' - CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)');"
echo.

:: 5. 메인 프로그램 실행
echo [START] 프로그램을 시작합니다...
set PYTHONIOENCODING=utf-8
python -m ui.main_window

:: 6. 종료 처리
if errorlevel 1 (
    echo.
    echo [ERROR] 프로그램이 에러와 함께 종료되었습니다.
    echo 위 로그를 캡처하여 디버깅하세요.
    pause
) else (
    echo.
    echo [INFO] 정상적으로 종료되었습니다.
    timeout /t 3 > nul
)

deactivate
endlocal