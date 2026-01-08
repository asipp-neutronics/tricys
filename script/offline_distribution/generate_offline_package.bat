@echo off
setlocal

echo ========================================================
echo Tricys Offline Package Generator
echo ========================================================

:: Updated PROJECT_ROOT to go up two levels
set "PROJECT_ROOT=%~dp0..\.."
set "DIST_DIR=%PROJECT_ROOT%\dist_tricys"
set "DIST_PACKAGES=%DIST_DIR%\packages"
set "DIST_SRC=%DIST_DIR%\src"

:: 1. Prepare Directories
echo [INFO] Cleaning up old distribution if exists...
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
mkdir "%DIST_DIR%"
mkdir "%DIST_PACKAGES%"
mkdir "%DIST_SRC%"

:: 2. Download Dependencies
echo [INFO] Downloading dependencies to %DIST_PACKAGES%...
echo [INFO] This makes sure we get binaries compatible with THIS machine's OS and Python version.
pip download -d "%DIST_PACKAGES%" -r "%PROJECT_ROOT%\requirements.txt"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download dependencies. Check your internet connection.
    pause
    exit /b 1
)

:: 3. Clone Source Code from Git
echo [INFO] Cloning source code from https://github.com/asipp-neutronics/tricys.git...
git clone https://github.com/asipp-neutronics/tricys.git "%DIST_SRC%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to clone repository. Check your internet connection or git installation.
    pause
    exit /b 1
)

:: 3.1 Overwrite with local requirements.txt (to match downloaded packages)
echo [INFO] Copying local requirements.txt to ensure version match...
copy "%PROJECT_ROOT%\requirements.txt" "%DIST_SRC%\" /Y >nul

:: Note: We are using the 'purified' code from Git, effectively ignoring local code changes.


:: 4. Copy Installer Script
echo [INFO] Copying install script...
copy "%~dp0install_offline.bat" "%DIST_DIR%\" /Y >nul

:: 5. Generate Readme with Version Info
echo [INFO] Generating offline_readme.txt...
for /f "tokens=2" %%I in ('python --version') do set PYTHON_VER=%%I

(
echo Tricys Offline Distribution Package
echo ===================================
echo.
echo Generated on: %DATE% %TIME%
echo Source Python Version: %PYTHON_VER%
echo.
echo INSTRUCTIONS:
echo 1. Ensure the target machine has Python %PYTHON_VER% installed.
echo    (Minor version mismatch like 3.10.x vs 3.10.y is usually okay, but major.minor MUST match)
echo 2. Run 'install_offline.bat' as Administrator.
echo.
echo This package includes all dependencies pre-downloaded in the 'packages' folder.
) > "%DIST_DIR%\offline_readme.txt"

echo.
echo ========================================================
echo Package generation complete!
echo Location: %DIST_DIR%
echo ========================================================
pause
