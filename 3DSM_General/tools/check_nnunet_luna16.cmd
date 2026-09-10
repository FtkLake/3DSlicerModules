@echo off
rem LUNA16 (肺結節) モデルで 1 症例を推論し Dice を表示する (画面なし)
setlocal
if "%SLICER_HOME%"=="" set SLICER_HOME=%LOCALAPPDATA%\slicer.org\Slicer5.8.1
set SLICER=%SLICER_HOME%\Slicer.exe
set REPO=%~dp0..
set GS_MODEL=LUNA16
if "%GS_CT%"=="" set GS_CT=D:\AIProj\LUNA16_Proj\nnUNet_raw\Dataset100_LUNA16\imagesTr\LUNA16_0098_0000.nii.gz
if "%GS_GT%"=="" set GS_GT=D:\AIProj\LUNA16_Proj\nnUNet_raw\Dataset100_LUNA16\labelsTr\LUNA16_0098.nii.gz
"%SLICER%" --no-splash --no-main-window --testing --additional-module-paths "%REPO%\GeneralSegmentation" --python-script "%REPO%\tools\check_nnunet.py"
echo exit code: %ERRORLEVEL%
endlocal
