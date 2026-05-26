@echo off
echo ============================================
echo  Face Anti-Spoofing Project Setup
echo  Python 3.12 + venv + RTX 30xx + Windows
echo ============================================

REM Step 1: Create virtual environment
echo [1/5] Creating virtual environment...
py -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create venv.
    pause & exit /b 1
)

REM Step 2: Activate
echo [2/5] Activating environment...
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate venv.
    pause & exit /b 1
)

REM Step 3: Upgrade pip
echo [3/5] Upgrading pip...
py -m pip install --upgrade pip

REM Step 4: Install PyTorch with CUDA 11.8 (stable for RTX 30xx)
echo [4/5] Installing PyTorch 2.x with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ERROR: PyTorch install failed.
    pause & exit /b 1
)

REM Step 5: Install all dependencies
echo [5/5] Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency install failed.
    pause & exit /b 1
)

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo  To activate your environment next time:
echo      fas-env\Scripts\activate
echo.
echo  To verify GPU:
echo      python verify_gpu.py
echo.
echo  To start training:
echo      python src/training/train.py --config configs/resnet18.yaml
echo.
pause
