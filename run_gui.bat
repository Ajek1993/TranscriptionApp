@echo off
REM Launch the GUI with the project venv interpreter.
REM Running plain `python gui.py` picks up the system Python, which has no
REM whisperx installed - the interface starts but transcription then fails.

setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo.
    echo BLAD: Nie znaleziono srodowiska venv:
    echo   %VENV_PYTHON%
    echo.
    echo Utworz je poleceniami:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements-windows.txt
    echo   pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

echo Uruchamianie GUI (http://127.0.0.1:7860)...
echo Zamknij to okno lub nacisnij Ctrl+C, aby zatrzymac.
echo.

"%VENV_PYTHON%" gui.py

if errorlevel 1 (
    echo.
    echo GUI zakonczylo sie bledem - tresc powyzej.
    pause
)

endlocal
