@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Create it with: py -3.11 -m venv .venv
  echo Then activate it and run: pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
