@echo off
setlocal
cd /d "%~dp0"
title Jarvis-Q1.58-500M

:menu
cls
echo ==========================================
echo   JARVIS Q1.58 500M
echo ==========================================
echo   1. Chat (interactive)
echo   2. One-shot prompt
echo   3. Evaluate (holdout perplexity)
echo   4. Benchmark CUDA engine (tok/s)
echo   5. Rebuild CUDA engine
echo   6. Exit
echo   7. Train 10M tokens (50%% English / 50%% code)
echo   8. Resume training
echo ==========================================
set /p c=Select: 
if "%c%"=="1" python jarvis_engine\evaluate_jarvis.py --interactive --skip-train-eval & pause & goto menu
if "%c%"=="2" set /p p=Prompt: & call :oneshot & pause & goto menu
if "%c%"=="3" python jarvis_engine\evaluate_jarvis.py --skip-gen & pause & goto menu
if "%c%"=="4" python -c "from jarvis_engine.cuda_engine import eng;eng.init();eng.warmup(3);eng.capture_graph();print(eng.benchmark(50))" & pause & goto menu
if "%c%"=="5" pushd jarvis_engine\cuda_engine & python build_engine.py & popd & pause & goto menu
if "%c%"=="6" exit /b
if "%c%"=="7" (if not exist data\train_tokens.npy python jarvis_engine\prepare_data.py 10000000) & python jarvis_engine\train_cuda.py & pause & goto menu
if "%c%"=="8" python jarvis_engine\train_cuda.py resume & pause & goto menu
goto menu

:oneshot
python jarvis_engine\evaluate_jarvis.py --skip-train-eval --prompt "%p%"
exit /b
