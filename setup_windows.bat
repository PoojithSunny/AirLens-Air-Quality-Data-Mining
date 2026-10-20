@echo off
setlocal
cd /d "%~dp0"
py -3 -m venv .venv
if errorlevel 1 goto usepython
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup finished. Put the official UCI ZIP in data\raw and run run_airlens.bat.
pause
exit /b 0
:usepython
python -m venv .venv
if errorlevel 1 goto failed
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Setup finished. Put the official UCI ZIP in data\raw and run run_airlens.bat.
pause
exit /b 0
:failed
echo Python was not found. Install Python 3.11-3.13 and enable Add Python to PATH.
pause
exit /b 1
