@echo off
echo ========================================
echo     BOSS OCR PIPELINE
echo ========================================
echo.

if "%1"=="" (
    echo Starting interactive mode...
    python boss_pipeline.py --interactive
) else (
    echo Processing: %1
    python boss_pipeline.py %1
)

pause
