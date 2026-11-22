# Jarvis Q1.58-606M Workspace Guide

Last audited: 2026-10-27. This file records the current workspace layout, the active execution paths, local data/checkpoint state, build assumptions, and known mismatches. Treat source code as the authority for current behavior; research notes and older docs contain stale claims.

## Current state at a glance

- Project: a 606M-class GPT-2-tokenizer language model with associative linear attention, ternary projections, top-2 MoE, liquid-state fusion, and a custom native CUDA trainer.
- Hardware target: NVIDIA GeForce RTX 5070 Blackwell, SM120, 12 GB GDDR7, 48 SMs, 48 MB L2. The current machine build used Python 3.14, PyTorch 2.11.0+cu128, CUDA Toolkit 13.3, and MSVC 2022 Build Tools 14.44.
- The active trainer is jarvis_engine/train_cuda.py plus jarvis_engine/cuda_engine/. The Windows entry point is start.bat.
- Default pretraining data from start.bat is the 300M-token, 25% English / 75% permissively licensed code corpus. The launcher defaults to 400M token exposures (399,998,976 complete tokens), so that corpus is reshuffled for a partial second pass.
- The local native checkpoint is data/jarvis_cuda.ckpt, 6,659,323,396 bytes, header step 48,828. Its sidecar records training_core_version 4 and 400M exposure settings; its data SHA matches the current 300M corpus. The checkpoint has not received conversational instruction tuning.
- The 400M-exposure checkpoint has produced incoherent text and immediate end-of-turn on a greeting. This is a quality failure, not a throughput issue: pretraining loss and tokens/sec do not establish chat behavior. jarvis_agent.py now reports an explicit diagnostic instead of silently printing an empty response.
- A mixed response-only SFT dataset was built and validated locally, but has not been trained yet: 14,999,883 Python-code tokens + 4,999,980 English conversation tokens, 70,989 complete examples, and 11,460,597 assistant-response targets. The user's next quality step is start.bat > 5 > I; the launcher evaluates held-out loss before and after. No post-SFT fluency claim is established yet.
- The workspace has pre-existing modified and untracked files, including code, data, profiler, and .vibecheck state. Preserve them; do not assume a clean Git tree or discard changes.

## Architecture and training semantics

The intended model dimensions are 24 layers, width 1,024, 16 heads of 64 dimensions, 4 experts with top-2 routing, expert width 2,048, GPT-2 vocabulary 50,257, padded vocabulary 50,304, training sequence length 512, and attention chunk size 64.

Each block uses RMSNorm, QKV projections, ELU+1 feature maps, RoPE, causal associative linear attention, an output projection and residual, a noisy top-2 router, two active expert FFNs, and liquid-state fusion. Attention carries a per-head recurrent state; inference can persist attention and liquid states across calls. Training resets per-sequence state at each microbatch. The final head uses online two-pass cross-entropy; the CUDA kernel skips the last position of each independent sequence.

Ternary projections are trained through FP32 master weights and an STE. The production-quality path uses AbsMean-scaled ternary values; the strict literal round(clamp(W,-1,1)) mapping is an experimental mode. With the current std=0.02 Gaussian initialization, the literal mapping collapses nearly all projection weights to zero, so do not enable it for a quality run without a validated initialization. Paper-shaped routing noise defaults to sigma=0.1; set JARVIS_ROUTER_NOISE=0 to disable it.

The native engine uses BF16 activations, FP32 master weights and AdamW moments, and an FP8 cached-weight forward path through cuBLASLt. If cuBLASLt has no FP8 algorithm for a routed expert shape, the engine falls back to BF16. The optimizer uses beta1=0.9, beta2=0.999, epsilon=1e-8, weight decay 0.01, gradient clipping 1.0, and a GPU-resident warmup/cosine schedule.

Core v4 includes the core-v3 corrections: RMSNorm2 backward uses its saved residual; LSF decay is computed from a global variance reduction; the LSF membrane sequence is recomputed for BPTT and gradients flow through its variance-derived decay; top-2 balancing counts both selected assignments; reflective regularization contributes an analytic activation gradient; router noise is seeded per layer. Core v4 also supports response-only target masks and an LR schedule offset in the native cross-entropy/optimizer path. These changes alter training semantics. Do not resume a pre-v4 checkpoint.

### Parameter-count and weight-tying caveat

The docs disagree. README.md and several experiments report 606,391,704 total and 405,065,112 active parameters. architecture.md reports 606,448,657 total and 353,600,000 active; another research report uses 606,390,000 / 353,600,000. Do not repeat the conflicting active count as verified.

There is also a real implementation difference: the native CUDA trainer stores one padded token-embedding matrix and ties the LM head to it. The PyTorch Jarvis class has separate embedding and LM-head modules; evaluate_jarvis.py copies the native embedding into both during binary-checkpoint loading. The native trainable tensor set is therefore about 555M parameters before small scalar bookkeeping, while the PyTorch model count includes a separate output matrix. Reconcile this with the paper and public parameter claims before describing the native model as exactly 606M trainable parameters.

## Active execution paths

### Training

- train_cuda.py loads a uint16 token array, converts it to int32, divides it into batches, validates checkpoints and metadata, performs one pass, prints loss/EMA/throughput, and saves the native checkpoint every 500 updates and at completion.
- Default settings: B=8, T=512, accumulation=2, 8,192 tokens/update, LR=2e-5, sparse MoE enabled, router noise sigma=0.1. JARVIS_ACCUM_STEPS, JARVIS_LR, JARVIS_MOE_SPARSE, JARVIS_ROUTER_NOISE, JARVIS_PAPER_FIDELITY, JARVIS_TRAIN_DATA, and JARVIS_TRAIN_MASK control run behavior.
- A 300M-token pass is 36,621 optimizer updates at the default batch; 20M-token SFT is 2,441 updates. The last partial update is not used.
- Resume modes are fresh, resume, resume-data, resume-lr, and resume-sft. Dataset/config sidecars prevent accidental incompatible resumes. resume-sft permits the intended data, mask, LR, schedule, and schedule-offset changes while preserving Adam's step for bias correction; its LR schedule starts at SFT step 1.
- Sparse top-2 routing is the default. It compacts selected token rows and disables CUDA Graph capture. The current path copies expert counts to the host and synchronizes per layer during forward and backward recomputation; this can be a major training bottleneck. Dense mode is selected with JARVIS_MOE_SPARSE=0 and can use CUDA Graphs.
- start.bat menu: 1 agent, 2 one-shot prompt, 3 evaluation plus benchmark, 4 rebuild, 5 train/resume, 0 exit. Option 5 offers pretraining or instruction tuning. SFT checks the checkpoint core version, evaluates before/after, rebuilds the native extension, and backs up the foundation checkpoint and sidecar before modifying them.

### Data preparation and SFT

- jarvis_engine/prepare_data.py streams Hugging Face Parquet shards with fsspec and PyArrow, reads only required columns, reports progress, and writes a disk-backed uint16 GPT-2 token array with EOT boundaries and a .meta manifest. GitHub code rows are filtered to MIT, ISC, BSD-2/3-Clause, Apache-2.0, Unlicense, Zlib, or CC0-1.0.
- jarvis_engine/prepare_sft_data.py mixes the cached Apache-2.0 Python instruction data with reviewed English conversations from Apache-2.0 OASST1. It creates User:/Assistant: examples, a deterministic holdout, and a response-only uint8 loss mask. --mask-only derives a mask locally from an existing token array without downloading data.
- Masked loss treats the user prompt as context and scores assistant response tokens plus EOT. It is normalized over active targets across both accumulation microbatches. CUDA Graphs are disabled for this path.
- The older code-only SFT set contains 20M uint16 tokens, 72,007 train examples, 3,862 holdout examples, and 1,064,826 holdout tokens. The new assistant_sft_tokens set requests a 15M/5M code/chat mix and stops only at complete example boundaries; actual size can be short by less than one example per source. Its current files contain 19,999,863 uint16 tokens, a same-length uint8 response mask with 11,460,597 active targets, 70,989 complete train examples, and 4,792 holdout examples (1,328,942 holdout tokens). Metadata format is user-assistant-mix-v3.
- SFT data generation reuses the cached code tokens and mask, trims at an EOT boundary, streams the reviewed English OASST1 train split, and labels only assistant response tokens plus EOT. The dataset card identifies OASST1 as Apache-2.0. Incomplete final examples are skipped instead of being cut before their Assistant delimiter.
- Recommended next quality step: run core-v4 response-only SFT on the existing foundation checkpoint, then compare held-out loss and greeting/code generations. Do not expect a single masked step or low training loss alone to establish fluency.

### Inference and agent

- evaluate_jarvis.py loads the native binary checkpoint into the PyTorch Jarvis model, reports backend/device details, supports one-shot and interactive generation, and evaluates text holdouts. Native projection tensors are dequantized for the PyTorch path; the token embedding is copied into both the embedding and LM head.
- The evaluator currently constructs the model with max_seq_len=256 while training uses T=512. Verify long-prompt behavior before claiming inference supports the full training context.
- jarvis_model.py is the PyTorch/reference model. It supports a CUDA associative-attention extension, a CUDA sparse-MoE extension, and persistent recurrent state for streaming inference when enabled.
- jarvis_engine/jarvis_agent.py wraps the model with local file, Python, shell, and GPU-information tools. Generated Python/shell execution requires explicit session consent; do not remove that guard.
- Current inference quality is poor despite the 400M-exposure core-v4 checkpoint. Chat/Agent now reports when generation immediately returns EOT; actual fluent behavior still needs to be measured after SFT.

## Data and checkpoint files currently present

| Path | Current role/state |
| --- | --- |
| data/train_tokens.npy | 20M-token English-only FineWeb-Edu GPT-2 token array; uint16, about 40 MB. |
| data/train_tokens.meta | English-only source/count manifest. |
| data/train_tokens_300m_en25_code75.npy | 300M-token corpus; 75M English + 225M licensed code; uint16, about 600 MB. |
| data/train_tokens_300m_en25_code75.meta | Source, mixture, count, and license manifest. |
| data/code_instruction_sft_tokens.npy | 20M-token Python instruction SFT array; uint16, about 40 MB. |
| data/code_instruction_sft_tokens_mask.npy | uint8 response-only loss mask, same token length. |
| data/code_instruction_sft_tokens_holdout.txt | Deterministic held-out instruction examples for evaluation. |
| data/code_instruction_sft_tokens.meta | Source, license, token count, example counts, tokenizer, and mask format. |
| data/assistant_sft_tokens.npy | 19,999,863-token code/conversation SFT array; actual 15M/5M source mix, uint16. |
| data/assistant_sft_tokens_mask.npy | Same-length uint8 response-only target mask for assistant answers and EOT. |
| data/assistant_sft_tokens_holdout.txt | Deterministic held-out code and OASST1 conversation examples. |
| data/assistant_sft_tokens.meta | Requested/actual source token counts, OASST1 license/source, examples, and mask targets. |
| data/jarvis_cuda.ckpt | Native float32 checkpoint containing weights, optimizer moments, and step header; local file is about 6.66 GB. |
| data/jarvis_cuda.ckpt.meta.json | Dataset SHA, training settings, data cursor, and checkpoint step; currently core v4. |

The current checkpoint sidecar SHA exactly matches train_tokens_300m_en25_code75.npy. Checkpoints are written atomically by the native engine. The four-byte step header is followed by serialized FP32 arrays; preserve the sidecar with the checkpoint. Do not delete or overwrite the old checkpoint as part of documentation-only work.

## Repository map

| Path | Purpose and status |
| --- | --- |
| README.md | User-facing overview and quickstart. Updated 2026-10-27 to reflect verified current defaults. |
| architecture.md | Detailed architecture spec and parameter inventory. It is marked frozen by phases.md but its counts conflict with the model/trainer. |
| phases.md | Project roadmap; updated 2026-10-27. Phase statuses and throughput corrected. |
| Jarvis_Engine_Research_Paper.pdf / paper_extracted.md | Local paper and extracted text used for architecture review. |
| memory.md | This operational map and current-state note. |
| All info.txt, JARVIS_MASTER_CONTEXT.txt, JARVIS_WORKSPACE_EXHAUSTIVE_AUDIT.txt | Large combined/historical workspace notes. Cross-check them against current source before relying on claims. |
| AGENTS.md, CLAUDE.md, .cursorrules, .clinerules, .windsurfrules, .claude/, .cursor/, .agents/ | Agent and IDE instructions/configuration; not model runtime code. Follow repository instructions for tools and edits. |
| .github/ | GitHub repository configuration. |
| workflows/ | Six Markdown runbooks for bug fixes, audits, feature work, pre-commit, pre-deploy, and release preparation. |
| jarvis_engine/ | Active Python model, trainer, data preparation, evaluation, agent, diagnostics, and native extension source. |
| jarvis_engine/cuda_engine/ | Main native C++/CUDA trainer. Source includes full_engine orchestration, CUDA workspace/parameter allocation, bindings, cuBLASLt, attention, MoE, LSF, ternary, optimizer, runtime, and generic training kernels. build/ and .pyd files are generated artifacts. |
| jarvis_engine/optimizers/ | PyTorch optimizer experiments (Muon, hybrid, parameter groups); the native CUDA training path currently uses its own AdamW implementation. |
| jarvis_engine/utils/ | PyTorch/reference ternary operations and MoE routing utilities. |
| associative_attention_cuda/ | Standalone PyTorch custom extension and validator for associative attention. |
| sparse_model_cuda/ | Standalone PyTorch custom sparse-MoE extension and validator. |
| liquid_fusion_cuda/ | Standalone PyTorch custom liquid-state extension and validator. |
| runtime/ | Experimental streaming, weight packing, weight streaming, MoE-aware streaming, scheduling, and CUDA Graph runner prototypes. These are not the primary start.bat training path. |
| benchmarks/ | Six baseline/profile/attention/memory/runtime benchmark scripts plus profiles/. Results are hardware/build dependent. |
| benchmark_production_ab.py, reproducible_benchmark.py, verify_production_integration.py | Root-level production A/B, repeatable benchmark, and integration verification utilities. |
| tests/ | Correctness gate, CUDA training-step check, optimizer infrastructure, native A/B comparison, and a large multi-phase 3B proof/benchmark harness. Some require an SM120 GPU and substantial time/VRAM; inspect before running. |
| research/ | Five documents: architecture decisions, methodology, interaction matrix, decision report, and hub README. |
| experiments/ | 175 research scripts/reports/results, not automatically part of production. Categories include architecture_matrix (memory_v2, ternary_packed, VRAM budget, reports), optimizer_research, ternary_paper_ab, baselines, context length, data-cleaning/original-data, LR, and STE-normalization experiments. |
| runs/ | Generated local run output; currently graph.json. |
| prof1.nsys-rep, prof1.sqlite, prof1_kern.txt, benchmarks/profiles/ | Nsight Systems/profile artifacts, not model inputs. |
| .vibecheck/ | Generated score/provenance/sync state. Its modifications are tooling state, not training code. |
| .gitignore | Ignores checkpoints, common build/cache files, selected large corpus files, and runtime output. Check Git status before staging data; SFT files may appear untracked. |

The main native engine is built from its own CUDA sources. The three standalone extension folders are separate prototypes/validation targets and are not needed to build the main engine unless a Python model backend explicitly loads them.

## Research and benchmark records

- research/architecture_decisions.md distinguishes the paper baseline from variants; methodology.md specifies controls; interaction_matrix.md and decision_report.md summarize memory-mechanism experiments.
- experiments/ternary_paper_ab/ contains the literal paper-quantizer collapse analysis and A/B results. Read weight_scale_analysis.md and phase_ternary_paper_fidelity_report.md before changing quantizer defaults.
- experiments/architecture_matrix/ contains activation, MoE, memory, packed-ternary, optimizer, retrieval, and long-context variants plus JSON/Markdown reports. These are research candidates, not production settings.
- experiments/optimizer_research/ contains optimizer audits, memory estimates, parameter inventories, and bakeoff plans. Native production still uses AdamW.
- Historical fixed-OC 50-update comparisons found sparse routing around 8% faster than dense mode with about 154-199 MB less allocated memory and similar short-run losses. These are not convergence results.
- Earlier builds reported 23.90-23.94k tok/s after a routing-count optimization. That is not a valid current-build claim: the current dynamic sparse route synchronizes per layer, and the latest core corrections have not had a matched performance/quality benchmark.
- Earlier microbatch sweeps: B=8/accum=1 about 21.35k tok/s; B=4/accum=4 about 20.37k tok/s with lower activation/logit memory; B=8/accum=4 about 25.70k; B=8/accum=8 about 26.35k versus 23.79k at accum=2 on a matched token window. Larger accumulation changes optimizer updates/effective batch, so speed alone does not select a quality default.
- The L2 persisting access window is 8 MB while the runtime set-aside cap is 24 MB. An older 8 MB cap sweep showed no gain. Current v3 throughput should be remeasured.
- README.md has been updated with current defaults (B=8/T=512/accum=2, LR=2e-5, 17-20k tok/s). phases.md and architecture.md section 7 have also been corrected.

## Build and routine commands

The main engine targets native SM120 only. The tested local build used PyTorch cu128 with CUDA Toolkit 13.3; the toolkit/PyTorch mismatch required explicit opt-in. MSVC Build Tools and NVCC are required. There is no single top-level requirements.txt or pyproject.toml; Python dependencies are currently implicit in module imports and each extension setup.py.

- Preferred Windows launcher: start.bat. Menu option 4 rebuilds; option 5 trains/resumes.
- Main extension rebuild: use start.bat option 4. For a direct build, set JARVIS_ALLOW_CUDA_MISMATCH=1, change into jarvis_engine/cuda_engine, and run rtk python build_engine.py there (setup.py is resolved relative to the current directory).
- Prepare the 300M corpus: rtk python jarvis_engine/prepare_data.py --english-tokens 75000000 --code-tokens 225000000 --output data/train_tokens_300m_en25_code75.npy. The script reports shard/token progress and publishes the completed array and metadata atomically.
- Create only the SFT mask from existing tokens: rtk python jarvis_engine/prepare_sft_data.py --mask-only --output data/code_instruction_sft_tokens.npy.
- Run one pass directly: set JARVIS_TRAIN_DATA to the desired corpus, then run rtk python jarvis_engine/train_cuda.py fresh. Direct default data is data/train_tokens.npy; start.bat selects the 300M corpus.
- Evaluate a prompt: rtk python jarvis_engine/evaluate_jarvis.py --skip-train-eval --prompt "Write Python code that prints hello world." --max_new_tokens 128.
- Evaluate the SFT holdout: use evaluate_jarvis.py --skip-train-eval --skip-gen --eval-file data/code_instruction_sft_tokens_holdout.txt with num-windows and seq-len options.
- Syntax and diff checks: rtk python -m py_compile jarvis_engine/train_cuda.py jarvis_engine/prepare_data.py jarvis_engine/prepare_sft_data.py jarvis_engine/evaluate_jarvis.py; rtk git diff --check.
- Correctness suites: tests/test_correctness_gate.py and tests/test_optimizer_infrastructure.py are separate from GPU tests; tests/test_cuda_correctness_suite.py and tests/test_jarvis_3b_proof.py may allocate large GPU buffers or run long experiments. Do not run the 3B proof harness casually.

When an environment variable must be set in PowerShell, set it before launching Python; the start.bat route handles the normal settings. Prefix shell/terminal commands with rtk per AGENTS.md. Do not delete workspace folders or data as part of ordinary cleanup; generated artifacts and research results may be referenced by other scripts.

## Current known limitations and next gates

1. ✅ Foundation 300M-token corpus and core-v4 training run completed (step 48,828).
2. ✅ Loss evaluated on mixed corpus; SFT data prepared (assistant_sft_tokens.npy).
3. ✅ O(1) inference state caching implemented in generate_streaming() / evaluate_jarvis.py.
4. **Current action: Run response-only SFT** — `start.bat > 5 > I`. Evaluates held-out loss before/after; backs up foundation checkpoint before modifying.
5. After SFT: inspect prompt outputs ("hi", code generation). Do not claim chat quality from loss alone.
6. Reconcile the 606M count with native tied embeddings and synchronize README.md / architecture.md only when explicitly requested.
7. Benchmark sparse-vs-dense and microbatch options on the current build with fixed clocks and matched data/quality checks; current throughput values (17-20k tok/s) are measured but have not been re-benchmarked since core v4 corrections.
8. README.md has been updated. The Apache LICENSE file still does not exist in the repo root — verify before publishing.
9. Preserve the current user's uncommitted changes, existing checkpoint, local data, profiler outputs, and IDE state unless explicitly asked to change them.
