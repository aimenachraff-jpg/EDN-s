@echo off
title ALSAT-2B Super-Resolution
cd /d "%~dp0"
if not exist "experiments\checkpoints\edsr_small_phase5_best.pt" (
    echo ERROR: Model checkpoint not found.
    echo Expected: experiments\checkpoints\edsr_small_phase5_best.pt
    pause
    exit /b 1
)
python sr_app.py %*
if errorlevel 1 (
    echo.
    echo An error occurred. See above for details.
    pause
)