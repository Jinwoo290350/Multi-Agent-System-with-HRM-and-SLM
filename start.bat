@echo off
echo 🤖 Decision Agent - Complete Task Routing System
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create data directory
if not exist "data" (
    echo 📁 Creating data directory...
    mkdir data
)

REM Check if .env exists
if not exist ".env" (
    echo ⚠️  .env file not found. Creating template...
    echo Please edit .env file with your actual configuration.
    echo.
)

REM Start the application
echo 🚀 Starting Decision Agent...
python decision_agent_complete.py

pause
