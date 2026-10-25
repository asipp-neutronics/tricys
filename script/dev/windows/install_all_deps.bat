@echo off
setlocal EnableDelayedExpansion

for %%I in ("%~dp0\..\..\..") do set ROOT_DIR=%%~fI

call :detect_python || exit /b 1

cd /d "%ROOT_DIR%"
call git submodule sync --recursive
call git submodule update --init --recursive

set VENV_DIR=%ROOT_DIR%\.venv
if exist "%ROOT_DIR%\venv\Scripts\activate.bat" set VENV_DIR=%ROOT_DIR%\venv

if not exist "%VENV_DIR%\Scripts\activate.bat" (
  call %PYTHON_LAUNCHER% -m venv "%VENV_DIR%" || exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

call pip install -e ".[dev,docs]" || exit /b 1

if exist "%ROOT_DIR%\tricys_backend\requirements.txt" (
  call pip install -r "%ROOT_DIR%\tricys_backend\requirements.txt" || exit /b 1
)

if exist "%ROOT_DIR%\tricys_visual\package.json" (
  cd /d "%ROOT_DIR%\tricys_visual"
  call npm.cmd install || exit /b 1
)

if exist "%ROOT_DIR%\tricys_goview\package.json" (
  cd /d "%ROOT_DIR%\tricys_goview"
  call :detect_pnpm || exit /b 1
  call pnpm.cmd install || exit /b 1
)

cd /d "%ROOT_DIR%"
echo Dependencies installed successfully.
echo Python virtual environment: %VENV_DIR%
endlocal
goto :eof

:detect_pnpm
where pnpm >nul 2>nul
if errorlevel 1 (
  echo --^> pnpm not found. Installing pnpm globally...
  call npm.cmd install -g pnpm || exit /b 1
)
exit /b 0

:detect_python
where py >nul 2>nul
if not errorlevel 1 (
  py --version >nul 2>nul
  if not errorlevel 1 (
    set PYTHON_LAUNCHER=py
    exit /b 0
  )
)

where python >nul 2>nul
if not errorlevel 1 (
  python --version >nul 2>nul
  if not errorlevel 1 (
    set PYTHON_LAUNCHER=python
    exit /b 0
  )
)

echo Error: Python launcher not found. Install Python and ensure py or python is available in PATH.
exit /b 1