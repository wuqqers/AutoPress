@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv-tk\Scripts\python.exe"
if not exist "%PYTHON%" goto missing_python

if /i "%~1"=="--check" goto check_setup

"%PYTHON%" -m autopress.app
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo AutoPress kapanirken hata kodu: %APP_EXIT%
    echo Ayrintilar yukarida gorunuyorsa bu pencereyi kapatmadan inceleyin.
    pause
)
exit /b %APP_EXIT%

:check_setup
"%PYTHON%" -c "import PySide6, sounddevice, vosk; from autopress.app import MODEL; assert MODEL.is_dir(), f'Turkce model bulunamadi: {MODEL}'; print('AutoPress hazir. Python, arayuz kutuphanesi, ses kutuphaneleri ve Turkce model bulundu.'); print(MODEL)"
if errorlevel 1 goto check_failed
exit /b 0

:missing_python
echo AutoPress sanal ortami bulunamadi:
echo %PYTHON%
echo README.md dosyasindaki kurulum adimlarini uygulayin.
pause
exit /b 1

:check_failed
echo AutoPress hazirlik kontrolu basarisiz oldu.
pause
exit /b 1
