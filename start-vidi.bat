@echo off
REM Double-click launcher for Vidi.
REM Starts the backend and frontend detached, over HTTPS, and fixes the
REM certificate first if your LAN address has changed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-vidi.ps1" %*
echo.
pause
