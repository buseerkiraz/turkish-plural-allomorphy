@echo off
REM Windows equivalent of run_all.sh. Run from the folder containing this file.
REM Writes each stage's output to logs\<stage>.log as well as to the screen,
REM matching what run_all.sh does with tee.
setlocal
cd /d "%~dp0"

REM Switch the console to UTF-8 so the IPA in the output renders instead of
REM appearing as boxes. Harmless if it fails.
chcp 65001 >nul 2>&1

if not exist output mkdir output
if not exist logs mkdir logs

REM Prefer the py launcher, which every Windows Python installer registers,
REM and fall back to python on PATH.
set "PY=python"
where py >nul 2>&1
if not errorlevel 1 set "PY=py -3"
echo Using interpreter: %PY%

for %%S in (00_fetch_data 01_extract 02_phase0_audit 03_at_control ^
            04_circularity_audit 05_plural_check 06_features 07_nonce_items ^
            08_alcove_benchmark 09_alcove_turkish) do (
  echo.
  echo ==================================================================
  echo ^>^>^> %%S.py
  echo ==================================================================
  %PY% "src\%%S.py" > "logs\%%S.log" 2>&1
  type "logs\%%S.log"
  if errorlevel 1 (
    echo.
    echo STAGE %%S FAILED - stopping. Full output in logs\%%S.log
    exit /b 1
  )
)

echo.
echo DONE. Outputs in output\, full logs in logs\
dir /b output
endlocal
