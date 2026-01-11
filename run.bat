@echo off
setlocal enabledelayedexpansion

:: ------------------------------------------------------------------
:: [Settings] Define Project Variables
:: ------------------------------------------------------------------
set "PROJECT_NAME=ObjectDetailer_Ultimate"
set "VENV_DIR=venv"

:: NOTE: We use absolute path for python exec later to avoid 'call activate' issues
set "PYTHON_EXEC=python"

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"
:: [Dual GPU Optimization] Limit CPU core usage per worker to avoid bottlenecks
set "OMP_NUM_THREADS=1"

:: Prevent console encoding issues (UTF-8)
chcp 65001 > nul

echo ================================================================
echo  🚀 %PROJECT_NAME% Launcher
echo ================================================================

:: 1. Check and Create Virtual Environment (venv)
if not exist "%VENV_DIR%" (
    echo [INFO] Virtual environment folder '%VENV_DIR%' not found. Creating new one...
    %PYTHON_EXEC% -m venv %VENV_DIR%
    
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Check if Python 3.10+ is installed.
        pause
        exit /b
    )
    echo [INFO] Virtual environment created.
)

:: 2. Activate Virtual Environment
:: Instead of 'call activate', we point directly to the venv executable.
set "PYTHON_EXEC=%CD%\%VENV_DIR%\Scripts\python.exe"

:: 2.1 Verify Python Executable
if not exist "%PYTHON_EXEC%" (
    echo [ERROR] Python executable not found: %PYTHON_EXEC%
    pause
    exit /b
)

:: Update pip and setuptools (Ensure stability)
%PYTHON_EXEC% -m pip install --upgrade pip setuptools

:: Check PyTorch (CUDA 12.1) Installation
%PYTHON_EXEC% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing/Restoring PyTorch CUDA 12.1 environment...
    %PYTHON_EXEC% -m pip install "torch>=2.1.0" "torchvision>=0.16.0" --index-url https://download.pytorch.org/whl/cu121
)

:: 3. Install/Update Required Packages
if exist "requirements.txt" (
    echo [INFO] Checking library dependencies...
    :: -q option skips already installed items quietly
    %PYTHON_EXEC% -m pip install -r requirements.txt
    
    if errorlevel 1 (
        echo [WARNING] Error occurred during library installation. Check network connection.
        pause
    )

    :: MediaPipe Compatibility Patch (Runs once)
    if not exist "%VENV_DIR%\.mediapipe_fixed_v7" (
        echo [INFO] Applying MediaPipe compatibility patch for Windows... - v7
        :: Uninstall related packages to prevent conflicts (including fixed NumPy version)
        %PYTHON_EXEC% -m pip uninstall -y mediapipe protobuf numpy
        :: Install stable version (0.10.14) and runtime libraries
        %PYTHON_EXEC% -m pip install --no-cache-dir --force-reinstall "mediapipe==0.10.14" "protobuf<5" "numpy<2" "msvc-runtime"
        echo fixed > "%VENV_DIR%\.mediapipe_fixed_v7"
    )
) else (
    echo [WARNING] 'requirements.txt' not found. Skipping package installation.
)

:: 4. Run Main Program
echo.
echo [INFO] Launching main program main.py...
echo Using Python: %PYTHON_EXEC%
echo ----------------------------------------------------------------

%PYTHON_EXEC% main.py

:: 5. Post-Execution Handling
if errorlevel 1 (
    echo.
    echo ----------------------------------------------------------------
    echo [ERROR] The program exited abnormally.
    echo Please check the error message above.
) else (
    echo.
    echo [INFO] Program finished successfully.
)