@echo off

where python >nul 2>nul
if errorlevel 1 (
    echo Python is required
    exit /b 1
)

echo Creating venv...
python -m venv venv

echo Activating venv...
call venv\Scripts\activate

echo Upgrading pip...
pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

if not exist model (
    echo Creating TensorFlow model...
    python save_model.py
)

echo Starting API...
python AI-Server.py

