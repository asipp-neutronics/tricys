@echo off
setlocal

REM ------------------------------------------------------------------------------
REM Makefile equivalent for Windows Batch Script
REM
REM Usage: Makefile.bat <command>
REM
REM Available commands:
REM   install       Install the project in editable mode for regular use.
REM   dev-install   Install the project with all development dependencies.
REM   clean         Remove all build artifacts, cache files, and logs.
REM   lint          Check code style and potential errors (report only, do not modify).
REM   format        Automatically format and repair code.
REM   check         Combine commands: format first, then check to make sure the codebase is clean.
REM   test          Perform one-click tests.
REM   uninstall     Uninstall the project.
REM   reinstall     Re-install the project (clean and install).
REM   help          Show this help message.
REM ------------------------------------------------------------------------------

REM Go to the directory where the script is located
cd /d "%~dp0"

REM Check if a command is provided
if "%~1"=="" ( goto :help )

REM Route to the correct command
goto :%~1


:help
echo Usage: Makefile.bat ^<command^>
echo.
echo Available commands:
echo   install       Install the project in editable mode for regular use.
echo   dev-install   Install the project with all development dependencies.
echo   clean         Remove all build artifacts, cache files, and logs.
echo   lint          Check code style and potential errors (report only, do not modify).
echo   format        Automatically format and repair code.
echo   check         Combine commands: format first, then check to make sure the codebase is clean.
echo   test          Perform one-click tests.
echo   uninstall     Uninstall the project.
echo   reinstall     Re-install the project (uninstall , clean and install).
echo   docs-install  Install documentation dependencies.
echo   docs-serve    Serve documentation site locally for development.
echo   docs-build    Build documentation site.
echo   help          Show this help message.
goto :eof


:install
echo --^> Installing project in editable mode...
call pip install -e .
echo --^> Installation complete.
goto :eof


:dev-install
echo --^> Installing project with development dependencies...
call pip install -e ".[dev]"
call pre-commit install
echo --^> Development installation complete.
goto :eof

:docs-install
echo --^> Installing documentation dependencies...
call pip install -e ".[docs]"
echo --^> Documentation dependencies installed.
goto :eof

:docs-serve
echo --^> Starting local documentation server...
call mkdocs serve
goto :eof

:docs-build
echo --^> Building documentation...
call mkdocs build
goto :eof

:win-install
echo --^> Installing project with development dependencies...
call pip install -e ".[win]"
call pre-commit install
echo --^> Development installation complete.
goto :eof


:clean
echo --^> Cleaning up project...
REM Remove __pycache__ directories
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

REM Remove .pyc files
del /s /q *.pyc >nul 2>nul

REM Remove other cache, build, and log directories/files
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .ruff_cache   rmdir /s /q .ruff_cache
if exist build         rmdir /s /q build
if exist dist          rmdir /s /q dist
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d"
if exist .coverage     del /f /q .coverage
if exist temp          rmdir /s /q temp
if exist log           rmdir /s /q log
if exist results       rmdir /s /q results
if exist data          rmdir /s /q data

REM This is a bit more complex to translate directly, assuming test_* are directories in test/
REM A simple loop can handle it if the pattern is consistent.
for /d %%i in (test\test_*) do (
    if exist "%%i" rmdir /s /q "%%i"
)

echo --^> Cleanup complete.
goto :eof


:lint
echo --^> Checking code with Ruff...
call ruff check .
goto :eof


:format
echo --^> Formatting code with Black...
call black .
echo --^> Sorting imports and fixing code with Ruff...
call ruff check . --fix
echo --^> Code formatting complete.
goto :eof


:check
call :format
call :lint
echo --^> All checks passed!
goto :eof


:test
echo --^> Pytest Project...
call pytest -v test\.
goto :eof


:uninstall
echo --^> Uninstalling project...
call pip uninstall tricys -y
echo --^> Uninstallation complete.
goto :eof


:reinstall
echo --^> Re-installing project...
call :uninstall
call :clean
call :win-install
echo --^> Re-installation complete.
goto :eof


REM Fallback for unknown commands
echo Error: Unknown command "%~1".
goto :help
