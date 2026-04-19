@echo off
echo =========================================
echo   Pyxis One — Backend Setup
echo =========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)

:: Set API key
echo.
echo [2/3] Anthropic API Key Setup
echo Get your key from: https://console.anthropic.com/settings/keys
echo.
set /p APIKEY="Paste your Anthropic API key (sk-ant-...): "

if "%APIKEY%"=="" (
    echo ERROR: No key entered.
    pause
    exit /b 1
)

:: Write .env file
echo ANTHROPIC_API_KEY=%APIKEY% > .env
echo DATABASE_URL=sqlite+aiosqlite:///./pyxis.db >> .env
echo SECRET_KEY=pyxis-secret-%RANDOM%%RANDOM% >> .env
echo ENVIRONMENT=development >> .env
echo PORT=8000 >> .env
echo FRONTEND_URL=http://localhost:3000 >> .env

echo.
echo [3/3] Starting backend server...
echo.
echo Backend will run at: http://localhost:8000
echo Health check: http://localhost:8000/health
echo.
python main.py
