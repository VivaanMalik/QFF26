@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv. Install Python 3.10+ from python.org and retry.
    exit /b 1
  )
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Starting Syndrome Siege on Windows...
streamlit run app.py --server.headless false
