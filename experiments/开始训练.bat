@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PINN_PYTHON=D:\Aanconda3\envs\pinn\python.exe"

if not exist "%PINN_PYTHON%" (
    echo Cannot find pinn Python: %PINN_PYTHON%
    exit /b 2
)

pushd "%~dp0.."
"%PINN_PYTHON%" "%~dp0run_scale_study.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
