@echo off
echo 🎯 STARTING 3D GENERATION WITH D: DRIVE CACHE
echo ================================================

REM Set Hugging Face cache to D: drive to prevent C: drive overflow
set HF_HOME=D:\huggingface_cache
set TRANSFORMERS_CACHE=D:\huggingface_cache
set HF_DATASETS_CACHE=D:\huggingface_cache
set TORCH_HOME=D:\huggingface_cache\torch

echo ✅ Cache directories set to D: drive:
echo    HF_HOME=%HF_HOME%
echo    TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%
echo    HF_DATASETS_CACHE=%HF_DATASETS_CACHE%
echo    TORCH_HOME=%TORCH_HOME%
echo.

REM Create cache directories if they don't exist
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"

echo 🚀 Starting 3D generation...
echo Prompt: %1
echo Max steps: %2

REM Activate virtual environment
call threestudio_env\Scripts\activate.bat

REM Run the 3D generation
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="%1" trainer.max_steps=%2 data.batch_size=1

echo.
echo ✅ 3D Generation Complete!
pause
