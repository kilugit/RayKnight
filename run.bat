@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found at venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Starting RayKnight Chess...
call venv\Scripts\python.exe -m chess_ai.gui.app %*
endlocal
