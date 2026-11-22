#!/usr/bin/env bash
# ==============================================================================
# Jarvis-Q1.58-606M Universal Launcher (Linux / macOS / WSL)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export JARVIS_ACCUM_STEPS="${JARVIS_ACCUM_STEPS:-2}"
export JARVIS_LR="${JARVIS_LR:-2e-5}"
export JARVIS_ROUTER_NOISE="${JARVIS_ROUTER_NOISE:-0.1}"
export JARVIS_TRAIN_TOKENS="${JARVIS_TRAIN_TOKENS:-400000000}"
export JARVIS_DEVICE="${JARVIS_DEVICE:-auto}"
export JARVIS_THERMAL_GUARD="${JARVIS_THERMAL_GUARD:-1}"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

show_model_status() {
    if [ -f "data/jarvis_cuda.ckpt" ]; then
        echo "Model checkpoint: found (data/jarvis_cuda.ckpt)"
    else
        echo "Model checkpoint: MISSING (run option 5 to train fresh)"
    fi
}

require_checkpoint() {
    if [ -f "data/jarvis_cuda.ckpt" ]; then
        return 0
    fi
    echo "No trained checkpoint found. Choose option 5 to train first."
    return 1
}

ensure_data() {
    if [ -f "$JARVIS_TRAIN_DATA" ] && [ -f "data/train_tokens_300m_en25_code75.meta" ]; then
        return 0
    fi
    echo "Preparing 300M tokens: 75M English and 225M permissively licensed code..."
    $PYTHON_BIN jarvis_engine/prepare_data.py --english-tokens 75000000 --code-tokens 225000000 --output "$JARVIS_TRAIN_DATA"
}

ensure_sft_data() {
    if [ -f "$JARVIS_TRAIN_DATA" ] && [ -f "data/assistant_sft_tokens_holdout.txt" ] && [ -f "data/assistant_sft_tokens.meta" ]; then
        if [ -f "data/assistant_sft_tokens_mask.npy" ]; then
            return 0
        fi
        $PYTHON_BIN jarvis_engine/prepare_sft_data.py --mask-only --output "$JARVIS_TRAIN_DATA"
        return 0
    fi
    echo "Preparing 15M Python and 5M English conversation tokens..."
    $PYTHON_BIN jarvis_engine/prepare_sft_data.py --tokens 20000000 --chat-tokens 5000000 --output "$JARVIS_TRAIN_DATA"
}

run_fresh() {
    export JARVIS_TRAIN_DATA="data/train_tokens_300m_en25_code75.npy"
    echo "Training 300M corpus for $JARVIS_TRAIN_TOKENS token exposures..."
    ensure_data
    $PYTHON_BIN jarvis_engine/train_cuda.py fresh
}

run_sanity() {
    export JARVIS_TRAIN_DATA="data/train_tokens.npy"
    export JARVIS_TRAIN_TOKENS="20000000"
    echo "Running quick sanity training on 20M tokens dataset..."
    $PYTHON_BIN jarvis_engine/train_cuda.py fresh
}

run_resume() {
    if [ ! -f "data/jarvis_cuda.ckpt" ]; then
        echo "No checkpoint found at data/jarvis_cuda.ckpt. Run fresh training first."
        return 1
    fi
    read -rp "Resume on dataset cursor (R) or resume at LR 2e-5 (L)? [R/L]: " res_choice
    case "$res_choice" in
        [lL]*) export JARVIS_LR="2e-5"; $PYTHON_BIN jarvis_engine/train_cuda.py resume-lr ;;
        *) $PYTHON_BIN jarvis_engine/train_cuda.py resume-data ;;
    esac
}

run_sft() {
    if [ ! -f "data/jarvis_cuda.ckpt" ]; then
        echo "Instruction tuning requires an existing foundation checkpoint (data/jarvis_cuda.ckpt)."
        return 1
    fi
    export JARVIS_TRAIN_DATA="data/assistant_sft_tokens.npy"
    export JARVIS_TRAIN_MASK="data/assistant_sft_tokens_mask.npy"
    export JARVIS_TRAIN_TOKENS="20000000"
    export JARVIS_LR="1e-4"
    export JARVIS_MOE_SPARSE="0"
    ensure_sft_data
    echo "Baseline held-out loss before SFT:"
    $PYTHON_BIN jarvis_engine/evaluate_jarvis.py --skip-train-eval --skip-gen --eval-file "data/assistant_sft_tokens_holdout.txt" --num-windows 128 --seq-len 256 --device "$JARVIS_DEVICE" || true
    if [ ! -f "data/jarvis_cuda_pre_sft.ckpt" ]; then
        echo "Backing up foundation checkpoint to data/jarvis_cuda_pre_sft.ckpt..."
        cp "data/jarvis_cuda.ckpt" "data/jarvis_cuda_pre_sft.ckpt"
        if [ -f "data/jarvis_cuda.ckpt.meta.json" ]; then
            cp "data/jarvis_cuda.ckpt.meta.json" "data/jarvis_cuda_pre_sft.ckpt.meta.json"
        fi
    fi
    echo "Starting Instruction Tuning (resume-sft)..."
    $PYTHON_BIN jarvis_engine/train_cuda.py resume-sft
    echo "Held-out loss after SFT:"
    $PYTHON_BIN jarvis_engine/evaluate_jarvis.py --skip-train-eval --skip-gen --eval-file "data/assistant_sft_tokens_holdout.txt" --num-windows 128 --seq-len 256 --device "$JARVIS_DEVICE"
}

rebuild_engine() {
    (
        cd jarvis_engine/cuda_engine
        export JARVIS_ALLOW_CUDA_MISMATCH=1
        $PYTHON_BIN build_engine.py || echo "Engine rebuild failed."
    )
}

# Direct CLI dispatch
case "$1" in
    fresh|train) run_fresh; exit 0 ;;
    sanity) run_sanity; exit 0 ;;
    resume) run_resume; exit 0 ;;
    sft) run_sft; exit 0 ;;
    rebuild) rebuild_engine; exit 0 ;;
    eval) require_checkpoint && $PYTHON_BIN jarvis_engine/evaluate_jarvis.py --skip-gen; exit 0 ;;
    agent|chat) require_checkpoint && $PYTHON_BIN jarvis_engine/jarvis_agent.py; exit 0 ;;
esac

menu() {
    clear 2>/dev/null || true
    echo "================================================================="
    echo "                      JARVIS Q1.58 606M                          "
    echo "================================================================="
    show_model_status
    echo "MoE Router: Gumbel-STE Top-2 (sigma=$JARVIS_ROUTER_NOISE)"
    echo "Thermal guard: ON (GPU throttle 75C, abort 80C)"
    echo "Training microbatch: B=8, T=512, accumulation=$JARVIS_ACCUM_STEPS"
    echo "Training token budget: $JARVIS_TRAIN_TOKENS"
    echo "Base learning rate: $JARVIS_LR"
    if [ "$JARVIS_PAPER_FIDELITY" = "1" ]; then
        echo "Literal paper ternary ablation: ON (fresh run required)"
    fi
    echo ""
    echo "  1  Chat / Agent"
    echo "  2  One-shot prompt"
    echo "  3  Evaluate + benchmark"
    echo "  4  Rebuild CUDA engine (SM120)"
    echo "  5  Pre-train Fresh (300M Corpus, 400M Budget)"
    echo "  6  Pre-train Quick Sanity (20M Tokens)"
    echo "  7  Resume Training"
    echo "  8  Instruction Fine-Tuning (SFT)"
    echo "  0  Exit"
    echo ""
    read -rp "Option: " choice

    case "$choice" in
        1)
            require_checkpoint || { read -rp "Press Enter to continue..."; return; }
            $PYTHON_BIN jarvis_engine/jarvis_agent.py --device "$JARVIS_DEVICE"
            read -rp "Press Enter to return to menu..."
            ;;
        2)
            require_checkpoint || { read -rp "Press Enter to continue..."; return; }
            read -rp "Enter prompt: " user_prompt
            if [ -n "$user_prompt" ]; then
                $PYTHON_BIN jarvis_engine/evaluate_jarvis.py --skip-train-eval --prompt "$user_prompt" --max_new_tokens 256 --device "$JARVIS_DEVICE"
            fi
            read -rp "Press Enter to return to menu..."
            ;;
        3)
            require_checkpoint || { read -rp "Press Enter to continue..."; return; }
            $PYTHON_BIN jarvis_engine/evaluate_jarvis.py --skip-gen --device "$JARVIS_DEVICE"
            $PYTHON_BIN jarvis_engine/bench_engine.py
            read -rp "Press Enter to return to menu..."
            ;;
        4)
            rebuild_engine
            read -rp "Press Enter to return to menu..."
            ;;
        5)
            run_fresh
            read -rp "Press Enter to return to menu..."
            ;;
        6)
            run_sanity
            read -rp "Press Enter to return to menu..."
            ;;
        7)
            run_resume
            read -rp "Press Enter to return to menu..."
            ;;
        8)
            run_sft
            read -rp "Press Enter to return to menu..."
            ;;
        0)
            echo "Exiting."
            exit 0
            ;;
        *)
            ;;
    esac
}

while true; do
    menu
done
