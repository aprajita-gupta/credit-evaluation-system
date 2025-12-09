@echo off
REM Credit Evaluation System - Production Startup Script
REM This script starts the application in production mode (no debug)

echo.
echo ================================================================
echo   CREDIT EVALUATION SYSTEM - PRODUCTION MODE
echo ================================================================
echo.

REM Check if virtual environment is activated
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please activate your virtual environment.
    echo.
    echo Run: .\venv\Scripts\activate
    echo Then run this script again.
    pause
    exit /b 1
)

echo [1/3] Checking Python version...
python --version

echo.
echo [2/3] Verifying dependencies...
python -c "import flask, numpy, pandas, sklearn" 2>nul
if errorlevel 1 (
    echo ERROR: Missing dependencies. Installing...
    pip install -r requirements-minimal.txt
) else (
    echo All core dependencies are installed.
)

echo.
echo [3/3] Starting Flask application...
echo.
echo ================================================================
echo   Server will start at: http://localhost:5000
echo   Press CTRL+C to stop the server
echo ================================================================
echo.

REM Start with the clean app (no debug mode)
python app_clean.py

pause