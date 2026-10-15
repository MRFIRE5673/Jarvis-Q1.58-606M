@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"
title Jarvis-Q1.58-500M

:menu
cls
echo ===================================================
echo               JARVIS Q1.58 500M CONSOLE
echo ===================================================
echo   1. Agentic AI (Python, CMD, File I/O, GPU execution)
echo   2. Raw Chat (Streaming text completion)
echo   3. One-shot prompt
echo   4. Evaluate (Holdout perplexity)
echo   5. Benchmark CUDA engine (tok/s)
echo   6. Rebuild CUDA engine (C++/SM120)
echo   7. Train 10M tokens (50%% English / 50%% code)
echo   8. Resume training (CUDA graph)
echo   9. Exit
echo ===================================================
set /p c=Select option [1-9]: 
if "%c%"=="1" goto agent
if "%c%"=="2" goto chat
if "%c%"=="3" goto oneshot_prompt
if "%c%"=="4" python jarvis_engine\evaluate_jarvis.py --skip-gen & pause & goto menu
if "%c%"=="5" python jarvis_engine\bench_engine.py & pause & goto menu
if "%c%"=="6" (
    pushd jarvis_engine\cuda_engine
    python build_engine.py
    popd
    copy /y jarvis_engine\cuda_engine\native_cuda_engine*.pyd . > nul
    echo Engine rebuild complete.
    pause
    goto menu
)
if "%c%"=="7" (
    if not exist data\train_tokens.npy (
        echo Preparing 10M training tokens...
        python jarvis_engine\prepare_data.py 10000000
    )
    python jarvis_engine\train_cuda.py
    pause
    goto menu
)
if "%c%"=="8" python jarvis_engine\train_cuda.py resume & pause & goto menu
if "%c%"=="9" exit /b
goto menu

:agent
cls
python jarvis_engine\jarvis_agent.py
pause
goto menu

:chat
cls
python jarvis_engine\evaluate_jarvis.py --interactive --skip-train-eval
pause
goto menu

:oneshot_prompt
echo.
set /p p=Enter Prompt: 
if "%p%"=="" goto menu
echo.
python jarvis_engine\evaluate_jarvis.py --skip-train-eval --prompt "%p%"
echo.
pause
goto menu
