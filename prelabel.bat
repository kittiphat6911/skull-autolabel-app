@echo off
chcp 65001 >nul
rem Drag and drop video files, image folders or .zip files onto this .bat
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag video files / image folders / zip files onto prelabel.bat
  pause
  exit /b
)
"D:\Master-degree\Auto_Annotation_Skull\venv\Scripts\python.exe" prelabel_local.py %*
echo.
echo Finished. The *_prelabel.zip files are next to the original files.
pause
