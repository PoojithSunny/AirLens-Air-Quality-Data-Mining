@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo No virtual environment found. Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python run_pipeline.py
if errorlevel 1 (
  echo Pipeline returned an error. Read the terminal output and README troubleshooting section.
  pause
  exit /b 1
)
echo.
echo To launch the dashboard, run: .venv\Scripts\python.exe -m streamlit run app.py
pause
