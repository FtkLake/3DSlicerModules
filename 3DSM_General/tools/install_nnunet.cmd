@echo off
rem Slicer の Python (PythonSlicer.exe) に nnU-Net 推論用パッケージ (requirements.txt) を入れる。
rem Slicer が既定の場所に無いときは、先に  set SLICER_HOME=C:\path\to\Slicer 5.8.1  としてから実行する。
setlocal
if "%SLICER_HOME%"=="" set SLICER_HOME=%LOCALAPPDATA%\slicer.org\Slicer5.8.1
set PYTHON=%SLICER_HOME%\bin\PythonSlicer.exe
set REPO=%~dp0..
if not exist "%PYTHON%" (
    echo PythonSlicer.exe not found: %PYTHON%
    echo Set SLICER_HOME to the Slicer install folder and run again.
    exit /b 1
)
echo Installing into: %PYTHON%
"%PYTHON%" -m pip install -r "%REPO%\requirements.txt"
if errorlevel 1 (
    echo pip install failed.
    exit /b 1
)
echo.
echo Verifying ...
"%PYTHON%" -c "import torch, importlib.metadata as m; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print('nnunetv2', m.version('nnunetv2')); print('numpy', m.version('numpy'))"
echo exit code: %ERRORLEVEL%
endlocal
