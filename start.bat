@echo off
setlocal
cd /d "%~dp0"
title Jarvis-Q1.58-606M

if not defined JARVIS_ACCUM_STEPS set "JARVIS_ACCUM_STEPS=2"
if not defined JARVIS_LR set "JARVIS_LR=2e-5"
if not defined JARVIS_ROUTER_NOISE set "JARVIS_ROUTER_NOISE=0.1"
if not defined JARVIS_TRAIN_TOKENS set "JARVIS_TRAIN_TOKENS=400000000"
if not defined JARVIS_THERMAL_GUARD set "JARVIS_THERMAL_GUARD=1"

rem Direct command line argument dispatch
if /i "%~1"=="fresh"   goto train_fresh
if /i "%~1"=="train"   goto train_fresh
if /i "%~1"=="sanity"  goto train_sanity
if /i "%~1"=="resume"  goto train_resume
if /i "%~1"=="sft"     goto train_instruction
if /i "%~1"=="rebuild" goto rebuild
if /i "%~1"=="eval"    goto evaluate_benchmark
if /i "%~1"=="agent"   goto agent
if /i "%~1"=="chat"    goto agent

:menu
cls
echo =====================================================================
echo                       JARVIS Q1.58 606M
echo =====================================================================
call :show_model_status
echo MoE Router: Gumbel-STE Top-2 (sigma=%JARVIS_ROUTER_NOISE%)
echo Thermal guard: ON (GPU throttle 75C, abort 80C)
echo Training microbatch: B=8, T=512, accumulation=%JARVIS_ACCUM_STEPS%
echo Training token budget: %JARVIS_TRAIN_TOKENS%
echo Base learning rate: %JARVIS_LR%
if "%JARVIS_PAPER_FIDELITY%"=="1" echo Literal paper ternary ablation: ON (fresh run required)
echo.
echo  1 Chat / Agent
echo  2 One-shot prompt
echo  3 Evaluate + benchmark
echo  4 Rebuild CUDA engine (SM120)
echo  5 Pre-train Fresh (300M Corpus, 400M Budget)
echo  6 Pre-train Quick Sanity (20M Tokens)
echo  7 Resume Training
echo  8 Instruction Fine-Tuning (SFT)
echo  0 Exit
echo.
set "c="
set /p "c=Option: "
if not defined c exit /b 0

if "%c%"=="1" goto agent
if "%c%"=="2" goto oneshot_prompt
if "%c%"=="3" goto evaluate_benchmark
if "%c%"=="4" goto rebuild
if "%c%"=="5" goto train_fresh
if "%c%"=="6" goto train_sanity
if "%c%"=="7" goto train_resume
if "%c%"=="8" goto train_instruction
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
python jarvis_engine\evaluate_jarvis.py --skip-train-eval --prompt "%p%" --max_new_tokens 256
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

:train_fresh
set "JARVIS_TRAIN_DATA=data\train_tokens_300m_en25_code75.npy"
echo The 300M-token 25/75 corpus will be trained for %JARVIS_TRAIN_TOKENS% exposures.
call :ensure_data
if errorlevel 1 goto back
python jarvis_engine\train_cuda.py fresh
goto back

:train_sanity
set "JARVIS_TRAIN_DATA=data\train_tokens.npy"
set "JARVIS_TRAIN_TOKENS=20000000"
echo Running quick sanity training on 20M tokens dataset...
python jarvis_engine\train_cuda.py fresh
goto back

:train_resume
if not exist "data\jarvis_cuda.ckpt" (
    echo No existing checkpoint found at data\jarvis_cuda.ckpt.
    echo Please run a fresh training run first (option 5).
    goto back
)
choice /c RL /n /m "Resume on dataset cursor (R) or resume at LR 2e-5 (L)? "
if errorlevel 2 goto resume_lr
python jarvis_engine\train_cuda.py resume-data
goto back

:resume_lr
set "JARVIS_LR=2e-5"
python jarvis_engine\train_cuda.py resume-lr
goto back

:train_instruction
if not exist "data\jarvis_cuda.ckpt" (
    echo Instruction tuning requires an existing native CUDA checkpoint (data\jarvis_cuda.ckpt).
    goto back
)
set "SAVED_JARVIS_TRAIN_DATA=%JARVIS_TRAIN_DATA%"
set "SAVED_JARVIS_LR=%JARVIS_LR%"
set "SAVED_JARVIS_TRAIN_MASK=%JARVIS_TRAIN_MASK%"
set "SAVED_JARVIS_TRAIN_TOKENS=%JARVIS_TRAIN_TOKENS%"
set "JARVIS_TRAIN_DATA=data\assistant_sft_tokens.npy"
set "JARVIS_TRAIN_MASK=data\assistant_sft_tokens_mask.npy"
set "JARVIS_TRAIN_TOKENS=20000000"
set "JARVIS_LR=1e-4"
set "JARVIS_MOE_SPARSE=0"
call :ensure_sft_data
if errorlevel 1 goto instruction_done
echo Baseline held-out conversation-and-code loss:
python jarvis_engine\evaluate_jarvis.py --skip-train-eval --skip-gen --eval-file "data\assistant_sft_tokens_holdout.txt" --num-windows 128 --seq-len 256
if errorlevel 1 goto instruction_done
echo.
echo Fine-tuning on 15M Python and 5M English conversation tokens...
if not exist "data\jarvis_cuda_pre_sft.ckpt" (
    echo Backing up foundation checkpoint before SFT.
    copy /b "data\jarvis_cuda.ckpt" "data\jarvis_cuda_pre_sft.ckpt" >nul
    if exist "data\jarvis_cuda.ckpt.meta.json" (
        copy /b "data\jarvis_cuda.ckpt.meta.json" "data\jarvis_cuda_pre_sft.ckpt.meta.json" >nul
    )
)
python jarvis_engine\train_cuda.py resume-sft
if errorlevel 1 goto instruction_done
echo.
echo Held-out conversation-and-code loss after fine-tuning:
python jarvis_engine\evaluate_jarvis.py --skip-train-eval --skip-gen --eval-file "data\assistant_sft_tokens_holdout.txt" --num-windows 128 --seq-len 256

:instruction_done
set "JARVIS_TRAIN_DATA=%SAVED_JARVIS_TRAIN_DATA%"
set "JARVIS_LR=%SAVED_JARVIS_LR%"
set "JARVIS_TRAIN_MASK=%SAVED_JARVIS_TRAIN_MASK%"
set "JARVIS_TRAIN_TOKENS=%SAVED_JARVIS_TRAIN_TOKENS%"
set "JARVIS_MOE_SPARSE="
goto back

:ensure_sft_data
if not exist "%JARVIS_TRAIN_DATA%" goto prepare_sft_data
if not exist "data\assistant_sft_tokens_holdout.txt" goto prepare_sft_data
if not exist "data\assistant_sft_tokens.meta" goto prepare_sft_data
if exist "data\assistant_sft_tokens_mask.npy" exit /b 0
python jarvis_engine\prepare_sft_data.py --mask-only --output "%JARVIS_TRAIN_DATA%"
if errorlevel 1 goto prepare_sft_data
exit /b 0

:prepare_sft_data
echo Preparing 15M Python and 5M English conversation tokens...
python jarvis_engine\prepare_sft_data.py --tokens 20000000 --chat-tokens 5000000 --output "%JARVIS_TRAIN_DATA%"
if errorlevel 1 exit /b 1
exit /b 0

:ensure_data
if not exist "%JARVIS_TRAIN_DATA%" goto prepare_data
if not exist "data\train_tokens_300m_en25_code75.meta" goto prepare_data
exit /b 0

:prepare_data
echo Preparing 300M tokens: 75M English and 225M permissively licensed code...
python jarvis_engine\prepare_data.py --english-tokens 75000000 --code-tokens 225000000 --output "%JARVIS_TRAIN_DATA%"
if errorlevel 1 exit /b 1
exit /b 0

:show_model_status
if exist "data\jarvis_cuda.ckpt" goto checkpoint_ready
echo Model checkpoint: MISSING (run option 5 to train fresh)
exit /b 0

:checkpoint_ready
echo Model checkpoint: found (data\jarvis_cuda.ckpt)
exit /b 0

:require_checkpoint
if exist "data\jarvis_cuda.ckpt" exit /b 0
echo No trained checkpoint found. Choose option 5 to train first.
exit /b 1

:back
if "%~1"=="" pause
goto menu
