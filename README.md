# Jarvis Q1.58-606M

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
- **Native CUDA C++ Engine:** cuBLASLt with BF16 activations, cached FP8 weights, and FP32 optimizer state. Training defaults to route-compacted top-2 MoE; that path disables CUDA Graphs and synchronizes to obtain each expert batch size, so measure throughput on the current build before comparing it with dense graph mode.
- **Paper-fidelity router:** Set JARVIS_PAPER_FIDELITY=1 to enable Gaussian (σ=0.1) router noise before top-2 selection.
- **Checkpoints & Training:** This repository contains the architecture, CUDA kernels, and training engine. It does **not** ship with a pretrained dialogue model. Train with `start.bat` option 5 or `python jarvis_engine/train_cuda.py`. A 606M model requires hundreds of millions to billions of tokens to develop natural English fluency and code synthesis.

---

## Quickstart Console (`start.bat`)

Run `start.bat` on Windows to access the interactive menu:

```text
===================================================
              JARVIS Q1.58 606M CONSOLE
===================================================
  1. Chat / Agent
  2. One-shot prompt
  3. Evaluate + benchmark
  4. Rebuild CUDA engine
  5. Train / resume
  0. Exit
===================================================
```

---

## Components

### 1. Agentic AI Terminal (`jarvis_engine/jarvis_agent.py`)
Provides local tool execution and reasoning:
- **`run_python` / `run_cmd`:** Execute generated Python or shell commands only after an explicit session consent prompt; commands run with the current Windows account's permissions.
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
- **jarvis_engine/prepare_data.py:** Streams selected Hugging Face Parquet shards into GPT-2 token arrays. The CLI defaults to 20M English tokens; start.bat prepares the 300M-token mix (75M FineWeb-Edu English, 225M license-filtered GitHub Code) in data/train_tokens_300m_en25_code75.npy.
- **`jarvis_engine/train_cuda.py`:** Executes training passes over the token array with live EMA loss, throughput telemetry, and periodic checkpointing.
- The native runner uses `B=8`, `T=512`, `accum=2` (8,192 tokens/update), FP32 AdamW state, FP8 weight caches, and a GPU-resident linear-warmup/cosine learning-rate schedule. The default LR is `1e-4`; override it with `JARVIS_LR`. Sparse top-2 routing is the training default; use `JARVIS_MOE_SPARSE=0` for dense graph replay.

```bash
# Tokenize the CLI's default 20M English-only corpus
python jarvis_engine/prepare_data.py 20000000

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
Jarvis-Q1.58-606M/
├── jarvis_engine/
│   ├── jarvis_model.py             # 606M PyTorch model definition (RoPE, MoE, LSF, Ternary)
│   ├── jarvis_agent.py             # Agentic AI terminal with local tool execution
│   ├── evaluate_jarvis.py          # Interactive CLI and holdout evaluation
│   ├── train_cuda.py               # Native CUDA training runner
│   ├── prepare_data.py             # FineWeb-Edu tokenization pipeline
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
