@echo off
setlocal
cd /d "%~dp0"
title Jarvis-Q1.58-606M
if not defined JARVIS_ACCUM_STEPS set "JARVIS_ACCUM_STEPS=2"

:menu
cls
echo JARVIS Q1.58 606M
echo.
call :show_model_status
echo Training microbatch: B=8, T=512, accumulation=%JARVIS_ACCUM_STEPS%
echo.
echo  1 Chat / Agent
echo  2 One-shot prompt
echo  3 Evaluate + benchmark
echo  4 Rebuild CUDA engine
echo  5 Train / resume
echo  0 Exit
echo.
set "c="
set /p "c=Option: "
if not defined c exit /b 0

if "%c%"=="1" goto agent
if "%c%"=="2" goto oneshot_prompt
if "%c%"=="3" goto evaluate_benchmark
if "%c%"=="4" goto rebuild
if "%c%"=="5" goto train
if "%c%"=="0" exit /b 0
goto menu

:agent
call :require_checkpoint
if errorlevel 1 goto back
python jarvis_engine\jarvis_agent.py
goto back

:oneshot_prompt
call :require_checkpoint
if errorlevel 1 goto back
set /p "p=Enter prompt: "
if not defined p goto menu
python jarvis_engine\evaluate_jarvis.py --skip-train-eval --prompt "%p%"
set "p="
goto back

:evaluate_benchmark
call :require_checkpoint
if errorlevel 1 goto back
python jarvis_engine\evaluate_jarvis.py --skip-gen
python jarvis_engine\bench_engine.py
goto back

:rebuild
pushd jarvis_engine\cuda_engine
set "JARVIS_ALLOW_CUDA_MISMATCH=1"
python build_engine.py
if errorlevel 1 goto rebuild_failed
popd
set "JARVIS_ALLOW_CUDA_MISMATCH="
echo Engine rebuild complete.
goto back

:rebuild_failed
popd
set "JARVIS_ALLOW_CUDA_MISMATCH="
echo Engine rebuild failed.
goto back

:train
set "JARVIS_TRAIN_DATA=data\train_tokens_300m_en25_code75.npy"
if not exist "data\jarvis_cuda.ckpt" (
    echo Starting from random weights with 300M tokens: 75M English and 225M code.
    echo.
)
call :ensure_data
if errorlevel 1 goto back
if not exist "data\jarvis_cuda.ckpt" goto train_fresh
choice /c RF /n /m "Resume current model on the new mixed corpus (R) or train fresh (F)? "
if errorlevel 2 goto train_fresh
python jarvis_engine\train_cuda.py resume-data
goto back

:train_fresh
python jarvis_engine\train_cuda.py fresh
goto back

:ensure_data
if not exist "%JARVIS_TRAIN_DATA%" goto prepare_data
if not exist "data\train_tokens_300m_en25_code75.meta" goto prepare_data
findstr /x /c:"mode=english-code-mix" data\train_tokens_300m_en25_code75.meta >nul 2>nul || goto prepare_data
findstr /x /c:"tokens=300000000" data\train_tokens_300m_en25_code75.meta >nul 2>nul || goto prepare_data
findstr /x /c:"english_tokens=75000000" data\train_tokens_300m_en25_code75.meta >nul 2>nul || goto prepare_data
findstr /x /c:"code_tokens=225000000" data\train_tokens_300m_en25_code75.meta >nul 2>nul || goto prepare_data
exit /b 0

:prepare_data
echo Preparing 300M tokens: 75M English and 225M permissively licensed code...
python jarvis_engine\prepare_data.py --english-tokens 75000000 --code-tokens 225000000 --output "%JARVIS_TRAIN_DATA%"
if errorlevel 1 exit /b 1
exit /b 0

:show_model_status
if exist "data\jarvis_cuda.ckpt" goto checkpoint_ready
if exist "ckpt_step_0004209.pt" goto checkpoint_ready
if exist "data\ckpt_step_0004209.pt" goto checkpoint_ready
echo Model checkpoint: MISSING - chat/evaluation would use random weights.
exit /b 0

:checkpoint_ready
echo Model checkpoint: found.
exit /b 0

:require_checkpoint
if exist "data\jarvis_cuda.ckpt" exit /b 0
if exist "ckpt_step_0004209.pt" exit /b 0
if exist "data\ckpt_step_0004209.pt" exit /b 0
echo No trained checkpoint. Choose 5 to train first; the 300M English/code corpus will be prepared then.
exit /b 1

:back
pause
goto menu
