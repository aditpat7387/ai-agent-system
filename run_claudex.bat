@echo off
cd /d C:\Users\ASUS\Downloads\Claudex
call venv\Scripts\activate
python src/agents/orchestrator.py >> logs\orchestrator.log 2>&1
