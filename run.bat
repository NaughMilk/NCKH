@echo off
echo 🚀 SDY Pipeline - Starting...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Run the main pipeline
python main.py %*

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ❌ Pipeline failed with error code %errorlevel%
    pause
)
