@echo off
start "CreditAlo API" cmd /k "call .venv\Scripts\activate && uvicorn creditalo.api:app --port 8000"
timeout /t 4 >nul
start "CreditAlo Dashboard" cmd /k "call .venv\Scripts\activate && streamlit run dashboard\app.py"
