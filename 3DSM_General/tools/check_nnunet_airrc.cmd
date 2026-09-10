@echo off
rem AirRC (気道・肺動静脈) モデルで fold_0 の検証症例を推論し Dice を表示する (画面なし)
setlocal
if "%SLICER_HOME%"=="" set SLICER_HOME=%LOCALAPPDATA%\slicer.org\Slicer5.8.1
set SLICER=%SLICER_HOME%\Slicer.exe
set REPO=%~dp0..
set GS_MODEL=AirRC
if "%GS_CT%"=="" set GS_CT=D:\AIProj\AirRC_Proj\nnUNet_raw\Dataset501_AirRC\imagesTr\AirRC_0009_0000.nii.gz
if "%GS_GT%"=="" set GS_GT=D:\AIProj\AirRC_Proj\nnUNet_raw\Dataset501_AirRC\labelsTr\AirRC_0009.nii.gz
"%SLICER%" --no-splash --no-main-window --testing --additional-module-paths "%REPO%\GeneralSegmentation" --python-script "%REPO%\tools\check_nnunet.py"
echo exit code: %ERRORLEVEL%
endlocal
