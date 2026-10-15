# Jarvis Q1.58-500M

Sub-quadratic 606M language model architecture combining ternary quantization (1.58-bit), chunked associative linear attention with RoPE, top-2 sparse Mixture-of-Experts (MoE), and liquid state fusion.

---

## Architecture Specifications

Verified parameter counts and architecture dimensions (606M scale):

| Specification | Value | Description |
| :--- | :---: | :--- |
| **Total Parameters** | **606,391,704** | Total model weights (embeddings + 24 blocks + LM head) |
| **Active Parameters / Token** | **405,065,112** | Active parameters per forward pass (~66.8% active via Top-2 MoE) |
| **Layers** | 24 | Transformer blocks with residual normalization |
| **Hidden Dimension ($d_{\text{model}}$)** | 1,024 | Internal representation width |
| **Attention Heads** | 16 | Associative attention heads (Head Dim: 64) |
| **Vocabulary Size** | 50,257 | GPT-2 BPE tokenizer vocabulary |
| **MoE Experts** | 4 | 4 routed experts per layer (Top-2 active per token) |
| **Weight Quantization** | 1.58-bit | Ternary weights $\{-1, 0, +1\}$ trained via Straight-Through Estimator (STE) |
| **Attention Mechanism** | Sub-quadratic | Chunked associative linear attention with Rotary Embeddings (RoPE) |
| **Recurrence** | Hybrid | Constant $O(1)$ memory state carry during autoregressive decoding |

---

## Current Status & Honest Grounding

- **Architecture & Kernels:** Fully implemented and passing 100% of correctness gates (forward/backward parity, STE gradient flow, causal invariance, state persistence, and numeric stability).
- **Native CUDA C++ Engine:** Custom cuBLASLt GEMM backend with FP8 cached activations, BF16 inputs, and FP32 accumulation under CUDA Graph capture. Benchmarked at **11,663+ tok/s** on an NVIDIA GeForce RTX 5070 (Blackwell SM120).
- **Checkpoints & Training:** This repository contains the architecture, CUDA kernels, and training engine. It does **not** ship with a pretrained dialogue model. Weights must be trained via the native training pipeline (`start.bat` Option 7 or `python jarvis_engine/train_cuda.py`). A 606M model requires hundreds of millions to billions of tokens to develop natural English fluency and code synthesis.

---

## Quickstart Console (`start.bat`)

Run `start.bat` on Windows to access the interactive menu:

```text
===================================================
              JARVIS Q1.58 500M CONSOLE
===================================================
  1. Agentic AI (Python, CMD, File I/O, GPU execution)
  2. Raw Chat (Streaming text completion)
  3. One-shot prompt
  4. Evaluate (Holdout perplexity)
  5. Benchmark CUDA engine (tok/s)
  6. Rebuild CUDA engine (C++/SM120)
  7. Train 10M tokens (50% English / 50% code)
  8. Resume training (CUDA graph)
  9. Exit
===================================================
```

---

## Components

### 1. Agentic AI Terminal (`jarvis_engine/jarvis_agent.py`)
Provides local tool execution and reasoning:
- **`run_python`:** Executes generated Python code in a sandboxed subprocess and returns stdout/stderr.
- **`run_cmd`:** Runs Windows CLI and shell commands.
- **`read_file` / `write_file`:** Inspects and modifies files in the workspace.
- **`gpu_info`:** Queries live NVIDIA VRAM, compute usage, and memory reservations.

Launch via `start.bat` (Option 1) or directly:
```bash
python jarvis_engine/jarvis_agent.py
```

### 2. High-Throughput Native CUDA Engine (`jarvis_engine/cuda_engine/`)
Custom C++/CUDA training engine with zero Python overhead during execution steps:
- **cuBLASLt GEMM Engine:** Pre-quantized FP8 weights with interleaved MoE W1 $\to$ GELU $\to$ W2 execution to maximize L2 cache hit rate.
- **CUDA Graph Capture:** Replays the entire forward, backward, optimizer step, and ternary clipping in a single GPU submission.
- **Binary Checkpoint Format:** Fast serialization directly to/from native host/device memory.

To build the native CUDA engine:
```bash
cd jarvis_engine/cuda_engine
python build_engine.py
```

To run the standalone CUDA benchmark:
```bash
python jarvis_engine/bench_engine.py
```

### 3. Data Preparation & Training Pipeline
- **`jarvis_engine/prepare_data.py`:** Streams and tokenizes multi-domain datasets (50% FineWeb-Edu English + 50% CodeParrot clean Python) into 512-token interleaved blocks (`data/train_tokens.npy`).
- **`jarvis_engine/train_cuda.py`:** Executes training passes over the token array with live EMA loss, throughput telemetry, and periodic checkpointing.

```bash
# Tokenize 10M tokens
python jarvis_engine/prepare_data.py 10000000

# Train with native CUDA engine
python jarvis_engine/train_cuda.py

# Resume training from existing checkpoint
python jarvis_engine/train_cuda.py resume
```

### 4. Interactive Evaluation Suite (`jarvis_engine/evaluate_jarvis.py`)
- Real-time streaming autoregressive decoding using $O(1)$ associative state caching.
- Perplexity and decoupled cross-entropy evaluation on holdout corpora.

```bash
# Streaming interactive session
python jarvis_engine/evaluate_jarvis.py --interactive

# Single prompt completion
python jarvis_engine/evaluate_jarvis.py --prompt "def fibonacci(n):" --max_new_tokens 64

# Holdout perplexity evaluation
python jarvis_engine/evaluate_jarvis.py --eval-file fresh_holdout.txt --num-windows 50
```

---

## Verification & Correctness Tests

Run the regression suite covering forward/backward parity, STE quantization gradient flow, causal invariance, state persistence, and numeric stability:

```bash
python tests/test_correctness_gate.py
```

---

## Repository Structure

```
Jarvis-Q1.58-500M/
├── jarvis_engine/
│   ├── jarvis_model.py             # 606M PyTorch model definition (RoPE, MoE, LSF, Ternary)
│   ├── jarvis_agent.py             # Agentic AI terminal with local tool execution
│   ├── evaluate_jarvis.py          # Interactive CLI and holdout evaluation
│   ├── train_cuda.py               # Native CUDA training runner
│   ├── prepare_data.py             # FineWeb-Edu + CodeParrot tokenization pipeline
│   ├── bench_engine.py             # Engine speed benchmark script
│   └── cuda_engine/                # Native C++/CUDA cuBLASLt training engine
│       ├── build_engine.py         # MSVC/NVCC build script
│       ├── cublaslt_engine.cu      # Fast cuBLASLt GEMM routines
│       ├── full_engine.cu          # Fused forward/backward/graph execution
│       └── core_kernels.cu         # Elementwise, norm, and ternary kernels
├── associative_attention_cuda/     # PyTorch custom CUDA extension for associative attention
├── sparse_model_cuda/              # PyTorch custom CUDA extension for sparse MoE
├── tests/                          # Rigorous numerical parity & correctness gate suites
├── experiments/                    # Research and ablation scripts
├── start.bat                       # One-click Windows console launcher
└── README.md                       # Repository documentation
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
