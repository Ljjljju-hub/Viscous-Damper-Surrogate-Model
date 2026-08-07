@echo off
setlocal
set "PINN_PYTHON=D:\Aanconda3\envs\pinn\python.exe"

if not exist "%PINN_PYTHON%" (
    echo Cannot find pinn Python: %PINN_PYTHON%
    exit /b 2
)

"%PINN_PYTHON%" "%~dp0run_remaining.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
