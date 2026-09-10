@echo off
rem GeneralSegmentation の自己テストを Slicer 上で実行する (画面なし)
setlocal
if "%SLICER_HOME%"=="" set SLICER_HOME=%LOCALAPPDATA%\slicer.org\Slicer5.8.1
set SLICER=%SLICER_HOME%\Slicer.exe
set REPO=%~dp0..
"%SLICER%" --no-splash --no-main-window --testing --additional-module-paths "%REPO%\GeneralSegmentation" --python-script "%REPO%\tools\run_tests.py"
echo exit code: %ERRORLEVEL%
endlocal
