@echo off
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="status" goto status
goto help

:start
echo Starting Milvus...
docker-compose up -d
echo.
echo Wait 30 seconds for Milvus to start...
timeout /t 30
echo Ready! Test with: python test_milvus_simple.py
goto end

:stop
echo Stopping Milvus...
docker-compose down
goto end

:status
echo Container Status:
docker ps --filter "name=milvus"
goto end

:help
echo Usage: milvus.bat [start^|stop^|status]
goto end

:end
