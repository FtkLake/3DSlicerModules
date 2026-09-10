@echo off
rem GeneralSegmentation モジュールを読み込んだ状態で 3D Slicer を起動する (設定変更なし)
setlocal
if "%SLICER_HOME%"=="" set SLICER_HOME=%LOCALAPPDATA%\slicer.org\Slicer5.8.1
set SLICER=%SLICER_HOME%\Slicer.exe
set REPO=%~dp0..
start "" "%SLICER%" --additional-module-paths "%REPO%\GeneralSegmentation"
endlocal
