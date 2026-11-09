@echo off
call .venv\Scripts\activate
echo API docs will be at http://127.0.0.1:8000/docs
uvicorn creditalo.api:app --host 127.0.0.1 --port 8000
