@echo off
setlocal
title CardioPulse
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "TFHUB_CACHE_DIR=%ROOT%.tfhub_cache"

echo ==================================================
echo   CardioPulse heart-sound monitor
echo ==================================================
echo.

if exist "%PY%" goto check_yamnet

echo First run: setting up the Python environment.
echo This needs an internet connection and can take 5 to 15 minutes.
echo.
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 -m venv "%VENV%" 2>nul
if exist "%PY%" goto install
py -3 -m venv "%VENV%"
if exist "%PY%" goto install

:use_python
python -m venv "%VENV%"
if exist "%PY%" goto install
echo.
echo ERROR: Python was not found.
echo Install Python 3.12 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation, then run this file again.
pause
exit /b 1

:install
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo.
    echo ERROR: installing the requirements failed. Check your internet connection and try again.
    rmdir /s /q "%VENV%"
    pause
    exit /b 1
)

:check_yamnet
if exist "%TFHUB_CACHE_DIR%" goto run
echo.
echo Downloading the YAMNet model once, so the dashboard can run without internet...
"%PY%" -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet saved.')"
if errorlevel 1 (
    echo.
    echo ERROR: could not download YAMNet. Connect to the internet for this first run, then try again.
    rmdir /s /q "%TFHUB_CACHE_DIR%" 2>nul
    pause
    exit /b 1
)

:run
echo.
echo --------------------------------------------------
echo  Before pressing "Connect" in the dashboard:
echo   1. Switch on the CardioPulse device.
echo   2. Join the Wi-Fi network  HeartMonitor_ESP32
echo      password: heartbeat123
echo  The dashboard opens at http://localhost:8501
echo  Close this window to stop CardioPulse.
echo --------------------------------------------------
echo.
cd /d "%ROOT%cardiopulse"
start "" /min cmd /c "timeout /t 10 >nul & start http://localhost:8501"
"%PY%" -m streamlit run dashboard_live.py
pause
