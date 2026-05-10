@echo off
cd backend
set PYTHONPATH=%CD%
call ..\.venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8001
