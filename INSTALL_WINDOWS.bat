@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3.10 -m venv .venv
    if not errorlevel 1 goto install
)

where python >nul 2>nul
if errorlevel 1 goto no_python
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)"
if errorlevel 1 goto no_python
python -m venv .venv
if errorlevel 1 goto failed

:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" CHECK_Module5_ENV.py
if errorlevel 1 goto failed
echo.
echo Installation complete. Double-click RUN_Module5.bat to start.
pause
exit /b 0

:no_python
echo Python 3.10 (64-bit) was not found. Install it, then run this file again.
pause
exit /b 1

:failed
echo Installation failed. Copy the error above when asking for help.
pause
exit /b 1
