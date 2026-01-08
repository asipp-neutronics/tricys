@echo off
setlocal

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "PACKAGES_DIR=%SCRIPT_DIR%packages"
set "SRC_DIR=%SCRIPT_DIR%src"

echo ========================================================
echo Tricys Offline Installer
echo ========================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python ^(version specified in offline_readme.txt^) and try again.
    pause
    exit /b 1
)

:: Check Python version (simple check, improve if needed)
for /f "tokens=2" %%I in ('python --version') do set PYTHON_VER=%%I
echo [INFO] Detected Python version: %PYTHON_VER%
echo [INFO] Please ensure this matches the required version in offline_readme.txt
echo.

echo [INFO] Installing dependencies from %PACKAGES_DIR%...
pip install --no-index --find-links="%PACKAGES_DIR%" -r "%SRC_DIR%\requirements.txt"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [INFO] Installing Tricys project...
cd /d "%SRC_DIR%"
pip install . --no-index --find-links="%PACKAGES_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Tricys.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo Installation Complete!
echo You can now run the project.
echo ========================================================
pause
