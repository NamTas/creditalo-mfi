@echo off
call .venv\Scripts\activate
python scripts\train.py %*
pause
