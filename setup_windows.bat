@echo off
echo ================================================
echo        MeetLingo - Windows Setup
echo   Real-time Voice Translation for Meetings
echo ================================================
echo.

:: Check Python version
python --version 2>NUL
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Upgrade pip
echo [1/5] Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA
echo [2/5] Installing PyTorch with CUDA support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install other dependencies
echo [3/5] Installing dependencies...
pip install -r requirements.txt

:: Install VibeVoice
echo [4/5] Installing Microsoft VibeVoice...
pip install git+https://github.com/microsoft/VibeVoice.git

:: Download voice prompts
echo [5/5] Downloading VibeVoice voice prompts...
if not exist "voices\streaming_model" mkdir voices\streaming_model
echo.
echo Please download voice files manually:
echo   1. Go to: https://github.com/microsoft/VibeVoice/tree/main/demo/voices/streaming_model
echo   2. Download .pt files to: voices\streaming_model\
echo.

echo.
echo ================================================
echo  Setup Complete!
echo ================================================
echo.
echo Next steps:
echo 1. Download voice .pt files to voices\streaming_model\
echo 2. Install VB-CABLE from: https://vb-audio.com/Cable/
echo 3. Copy env.example to .env and add your API keys
echo 4. Run: set HF_HUB_DISABLE_SYMLINKS_WARNING=1 ^&^& python main.py
echo.
pause

