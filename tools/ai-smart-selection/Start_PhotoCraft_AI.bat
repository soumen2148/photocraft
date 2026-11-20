@echo off
setlocal
set "PYW="
for %%P in (pythonw.exe) do set "PYW=%%~$PATH:P"
if not defined PYW (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*" "C:\Python3*") do (
        if exist "%%D\pythonw.exe" set "PYW=%%D\pythonw.exe"
    )
)
if not defined PYW (
    echo Python 3 with pythonw.exe was not found on PATH.
    pause
    exit /b 1
)
start "" "%PYW%" "%~dp0photocraft_ai_hook.py" --ensure-gui %*
