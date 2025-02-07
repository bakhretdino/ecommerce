@echo off
setlocal enabledelayedexpansion

REM Set the virtual environment directory name
set VENV_DIR=venv

REM Check if the virtual environment exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
) else (
    echo Virtual environment already exists.
)

REM Activate the virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate

REM Install dependencies
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found! Skipping dependency installation.
)