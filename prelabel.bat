@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=D:\Master-degree\Auto_Annotation_Skull\venv\Scripts\python.exe"
if not "%~1"=="" goto run_args

:ask
echo.
echo ============================================================
echo  Skull pre-label
echo  Drag a VIDEO, an IMAGE FOLDER or a ZIP into this window,
echo  then press Enter.  (Leave empty and press Enter to quit)
echo ============================================================
set "INPUT="
set /p "INPUT=File/folder: "
if not defined INPUT exit /b
"%PY%" prelabel_local.py %INPUT%
echo.
echo Done. The *_prelabel.zip file is next to the original file.
goto ask

:run_args
"%PY%" prelabel_local.py %*
echo.
echo Done. The *_prelabel.zip files are next to the original files.
pause
