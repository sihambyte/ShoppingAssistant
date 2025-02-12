@echo off
call venv\Scripts\activate
python -m backend.experiments
call deactivate 