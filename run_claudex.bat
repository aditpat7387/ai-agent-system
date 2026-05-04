@echo off
cd /d C:\Users\ASUS\Downloads\Claudex

if not exist logs mkdir logs

call venv\Scripts\activate.bat
python src\run_pipeline.py >> logs\pipeline.log 2>&1

exit /b %errorlevel%