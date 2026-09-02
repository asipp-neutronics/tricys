@echo off
setlocal

for %%I in ("%~dp0\..\..\..") do set ROOT_DIR=%%~fI
set PID_DIR=%ROOT_DIR%\.run

if not exist "%PID_DIR%" mkdir "%PID_DIR%"

call :check_port 8000 tricys_backend
if errorlevel 1 goto :eof

call :check_port 3020 tricys_goview
if errorlevel 1 goto :eof

call :check_port 5173 tricys_visual
if errorlevel 1 goto :eof

call :check_port 8050 tricys_hdf5
if errorlevel 1 goto :eof

set VENV_ACTIVATE=%ROOT_DIR%\.venv\Scripts\activate.bat
if not exist "%VENV_ACTIVATE%" set VENV_ACTIVATE=%ROOT_DIR%\venv\Scripts\activate.bat

set HDF5_SECRET=%HDF5_VISUALIZER_SECRET%
if "%HDF5_SECRET%"=="" set HDF5_SECRET=your-super-secret-key-change-in-production

set HDF5_CONTEXTS_DIR=%HDF5_CONTEXTS_DIR%
if "%HDF5_CONTEXTS_DIR%"=="" set HDF5_CONTEXTS_DIR=%ROOT_DIR%\tricys_backend\hdf5_contexts
if not exist "%HDF5_CONTEXTS_DIR%" mkdir "%HDF5_CONTEXTS_DIR%"

if exist "%VENV_ACTIVATE%" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', 'cd /d \"\"%ROOT_DIR%\"\" && call \"\"%VENV_ACTIVATE%\"\" && python -m uvicorn tricys_backend.main:app --host 0.0.0.0 --port 8000 --reload' -PassThru; Set-Content -Path '%PID_DIR%\tricys_backend.pid' -Value $proc.Id"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', 'cd /d \"\"%ROOT_DIR%\"\" && call \"\"%VENV_ACTIVATE%\"\" && tricys hdf5 --server-mode --host 0.0.0.0 --port 8050 --base-pathname /hdf5/ --secret \"\"%HDF5_SECRET%\"\" --context-dir \"\"%HDF5_CONTEXTS_DIR%\"\" --no-browser' -PassThru; Set-Content -Path '%PID_DIR%\tricys_hdf5.pid' -Value $proc.Id"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', 'cd /d \"\"%ROOT_DIR%\"\" && python -m uvicorn tricys_backend.main:app --host 0.0.0.0 --port 8000 --reload' -PassThru; Set-Content -Path '%PID_DIR%\tricys_backend.pid' -Value $proc.Id"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', 'cd /d \"\"%ROOT_DIR%\"\" && tricys hdf5 --server-mode --host 0.0.0.0 --port 8050 --base-pathname /hdf5/ --secret \"\"%HDF5_SECRET%\"\" --context-dir \"\"%HDF5_CONTEXTS_DIR%\"\" --no-browser' -PassThru; Set-Content -Path '%PID_DIR%\tricys_hdf5.pid' -Value $proc.Id"
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', 'cd /d \"\"%ROOT_DIR%\tricys_visual\"\" && npm.cmd run dev' -PassThru; Set-Content -Path '%PID_DIR%\tricys_visual.pid' -Value $proc.Id"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/k', 'cd /d \"\"%ROOT_DIR%\tricys_goview\"\" && pnpm.cmd run dev' -PassThru; Set-Content -Path '%PID_DIR%\tricys_goview.pid' -Value $proc.Id"

endlocal
goto :eof

:check_port
set PORT=%~1
set SERVICE_NAME=%~2

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  echo Error: port %PORT% is already in use, so %SERVICE_NAME% cannot start.
  echo Hint: if you previously started the Docker stack, run "docker compose down" first.
  echo Hint: if you previously started the local app stack, run "Makefile.bat app-stop" first.
  exit /b 1
)

exit /b 0
