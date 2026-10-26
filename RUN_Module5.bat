@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Dependencies are not installed. Run INSTALL_WINDOWS.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" CHECK_Module5_ENV.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto failed
exit /b 0

:failed
echo Module 5 could not start. Copy the error above when asking for help.
pause
exit /b 1
