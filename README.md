# Jarvis Q1.58-606M

An experimental Windows/CUDA language-model project with a native training engine, a PyTorch model/evaluator, associative attention, and sparse top-2 MoE. The project name says "606M"; the native trainer ties the token embedding and LM head into one matrix while the PyTorch model keeps them separate — so the effective trainable count differs slightly between the two. Treat "606M" as a project label until the two implementations are formally reconciled.

## Current state

This is a research implementation. The 400M-token core v4 foundation checkpoint (`data/jarvis_cuda.ckpt`, step 48,828) exists and has been validated for training correctness, but currently produces incoherent or empty output on conversational prompts. **The next quality step is instruction tuning (SFT)** via `start.bat > 5 > I`. The SFT dataset is already prepared (`data/assistant_sft_tokens.npy`, 15M code + 5M OASST1 conversation, ~11.5M response targets).

The main architecture settings are 24 blocks, width 1,024, 16 attention heads, a 50,257-token GPT-2 vocabulary, and four experts with top-2 routing. The design includes chunked associative linear attention with RoPE and liquid state fusion. The CUDA engine uses BF16 activations with FP32 master weights and optimizer state. FP8 weight caches are used for supported GEMMs; this does not mean the checkpoint is stored in FP8. On SM120, grouped expert GEMM is disabled and sparse FP8 fallback is used. Dynamic expert batches (sparse mode default) disable CUDA Graph capture.

## Start the console

On Windows, run `start.bat` from the repository directory:

```text
1  Chat / Agent
2  One-shot prompt
3  Evaluate + benchmark
4  Rebuild CUDA engine
5  Train / resume
0  Exit
```

Option 5 asks whether to pretrain (P) or instruction-tune (I). Instruction tuning evaluates the held-out loss before and after, backs up the foundation checkpoint, and runs the SFT pass. For a fresh pretrain run, choose P → F (fresh). The old checkpoint predates core v4; use fresh for a clean run.

## Training configuration (current defaults)

| Setting | Value |
|---|---|
| Micro-batch B | 8 |
| Sequence length T | 512 |
| Gradient accumulation | 2 |
| Tokens per optimizer step | 8,192 |
| Default LR | 2e-5 |
| Router noise σ | 0.1 |
| Default token budget | 400M (≈48,828 updates) |
| Measured throughput | 13,800–18,000 tok/s (RTX 5070, sparse mode; benchmarked: 13,869 tok/s) |
| Estimated wall time | ~6–8 hours |

Set environment variables before calling `start.bat` to override:

```bat
set JARVIS_TRAIN_TOKENS=300000000
set JARVIS_LR=2e-5
start.bat
```

## Data preparation

The console prepares the 300M mixed corpus if its array or metadata is missing. Before instruction tuning it evaluates the held-out mix, saves `data/jarvis_cuda_pre_sft.ckpt` as a restore point, runs SFT, then evaluates again. Manual commands:

```bat
python jarvis_engine/prepare_data.py --english-tokens 75000000 --code-tokens 225000000 --output data/train_tokens_300m_en25_code75.npy
python jarvis_engine/prepare_sft_data.py --tokens 20000000 --chat-tokens 5000000 --output data/assistant_sft_tokens.npy
```

Preparation streams selected Hugging Face Parquet shards and reports progress. The code filter accepts MIT, ISC, BSD-2-Clause, BSD-3-Clause, Apache-2.0, Unlicense, Zlib, and CC0-1.0. The conversational source is the English, reviewed, non-synthetic subset of OpenAssistant OASST1 (Apache-2.0). Check each dataset's source terms before redistribution.

## Build and run directly

```bat
cd jarvis_engine\cuda_engine
python build_engine.py
cd ..\..
python jarvis_engine\train_cuda.py fresh
python jarvis_engine\evaluate_jarvis.py --interactive
```

Local build environment: Windows, Python 3.14, PyTorch 2.11+cu128, CUDA Toolkit 13.3, MSVC 2022 14.44, RTX 5070 SM120. The console rebuild sets `JARVIS_ALLOW_CUDA_MISMATCH=1`.

For a prompt or holdout evaluation:

```bat
python jarvis_engine/evaluate_jarvis.py --prompt "Write a Python function that adds two numbers." --max_new_tokens 128
python jarvis_engine/evaluate_jarvis.py --skip-train-eval --skip-gen --eval-file data/assistant_sft_tokens_holdout.txt --num-windows 128 --seq-len 256
```

The evaluator uses `max_seq_len=256`; training uses T=512. Evaluate with consistent settings when comparing runs.

## Workspace map

```text
jarvis_engine/              Native trainer, PyTorch model, data prep, evaluation, agent
jarvis_engine/cuda_engine/  Native C++/CUDA engine and build script
associative_attention_cuda/ PyTorch associative-attention CUDA extension
sparse_model_cuda/          PyTorch sparse-MoE CUDA extension
liquid_fusion_cuda/         Liquid-state CUDA prototype
experiments/                Architecture, optimizer, ternary, and data experiments
research/                   Paper notes and architectural decision records
benchmarks/                 Benchmark scripts (results are hardware/build-specific)
tests/                      Correctness and regression checks
data/                       Local token arrays, masks, metadata, and checkpoints
memory.md                   Detailed workspace and operational notes (authoritative)
start.bat                   Windows menu and training launcher
```

## Known limitations

- The native trainer and PyTorch model have separate tok_emb and lm_head; checkpoint loading copies the native embedding into both. Parameter count differs slightly between the two implementations.
- Sparse top-2 routing uses dynamic expert batches and disables CUDA Graphs. The SM120 grouped expert GEMM path is unavailable in the local build.
- FP8 caches are a forward-pass optimization, not proof of FP8 or ternary optimizer updates.
- A low training loss does not guarantee useful language generation. Inspect generated samples after training and after SFT.
- No top-level LICENSE file is present. Verify repository licensing before publishing.

## Correctness checks

```bat
python tests/test_correctness_gate.py
```

Run after any change to model math or CUDA kernels.
