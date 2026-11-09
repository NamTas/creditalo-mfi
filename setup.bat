@echo off
echo === CreditAlo : one-time setup ===
python --version || (echo Python not found. Install Python 3.11 and tick "Add to PATH". & pause & exit /b 1)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
echo.
echo Setup finished. Next: double-click train.bat
pause
