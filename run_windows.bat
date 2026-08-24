@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher non trovato. Installa Python 3.11 e riprova.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto :error
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
pip install -r requirements.txt
if errorlevel 1 goto :error
python -m app
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Avvio non riuscito. Copia qui il messaggio di errore.
pause
exit /b 1
