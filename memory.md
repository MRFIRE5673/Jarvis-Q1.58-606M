# Jarvis Q1.58-606M Working Memory

## Current production path

- Target GPU: NVIDIA RTX 5070 Blackwell, SM120, 12 GB VRAM.
- Native engine: `jarvis_engine/cuda_engine/`; build with CUDA 12.8-compatible PyTorch and the installed CUDA toolkit.
- Forward path: BF16 activations with FP8 cached weight GEMMs through cuBLASLt.
- Cached FP8 GEMM falls back to BF16 when cuBLASLt has no algorithm for a routed expert batch shape; this keeps small expert batches valid.
- Training state: FP32 master weights and FP32 AdamW moments; ternary projections use AbsMean scaling with a straight-through estimator.
- Dense mode captures forward, backward, optimizer update, and re-quantization in a CUDA Graph; sparse top-2 training uses the direct route-compacted path.

## Training defaults

- Training corpus: data/train_tokens_300m_en25_code75.npy contains exactly 300M GPT-2 tokens (75M FineWeb-Edu English, 225M GitHub Code filtered to MIT/ISC/BSD/Apache/Unlicense/Zlib/CC0). prepare_data.py lists shard names once, streams only needed Parquet columns via ranged reads, reports progress, and atomically publishes the corpus plus .meta after completion. Code rows use the Parquet content field. This corpus is prepared but has not been trained on; it does not by itself guarantee fluent output.
- Batch: `B=8`, sequence length: `T=512`, accumulation: `2`.
- Effective tokens per optimizer update: `8,192`.
- Set `JARVIS_ACCUM_STEPS=1` for a 4,096-token update.
- Training runner defaults to sparse top-2 routing (`JARVIS_MOE_SPARSE=1`); set `JARVIS_MOE_SPARSE=0` to force dense CUDA Graph mode. Paired fixed-OC RTX 5070 runs on real token data and fresh random weights (50 updates × 8,192 tokens) averaged about 8% faster sparse than dense. Dense samples: 16.69–19.50k tok/s; sparse: 18.06–21.16k tok/s. Device memory use was 154–199 MB lower in sparse runs. Losses stayed close (dense final 6.386–6.390; sparse 6.386–6.409); short throughput/trajectory checks do not establish convergence. Absolute speed varied across runs; live clocks were not logged.
- L2 pin sweep: default set-aside cap is 24 MB, while the active persisting access window is 8 MB (`layer_x`). Reducing `JARVIS_L2_PIN_MB` to 8 MB did not improve throughput, so keep the default cap.
- Sparse routing now accumulates expert counts inside top-2 gating, removing the separate count-kernel launch. After rebuilding, two 50-update real-token runs measured 23.90–23.94k tok/s at B=8/T=512/accum=2. This is a current throughput result, not a controlled before/after attribution or a convergence check; the 1e-4 LR run's loss was effectively flat at five decimal places.
- Same-host microbatch sweep (50 measured updates, real tokens): B=8/accum=1 reached 21.35k tok/s; B=4/accum=4 reached 20.37k tok/s and roughly halved token-dependent activation/logit buffers. Keep B=8/accum=2 for throughput; B=4/accum=4 is the lower-memory fallback.
- B=8/accum=8 was also tested against B=8/accum=2 for equal 393,216-token windows: 26.35k vs 23.79k tok/s (+10.8%). Accum=8 performs 12 optimizer updates versus 48 at accum=2, so this is a throughput-only tradeoff with a 4× larger effective batch and different LR schedule; keep accum=2 as the quality-safe default until a matched quality run validates accum=8.
- B=8/accum=4 on the same 393,216-token window measured 25.70k tok/s (+8.0% vs accum=2, -2.5% vs accum=8). It performs 24 optimizer updates, making it a middle throughput/effective-batch setting; quality still needs a controlled validation run.
- AdamW: default `lr=1e-4` (`JARVIS_LR` override), `beta1=0.9`, `beta2=0.999`, `weight_decay=0.01`, gradient clip `1.0`. CUDA skips optimizer updates when the gradient norm is non-finite; training also checks returned loss. Checkpoint saves replace the previous file atomically, and loading rejects truncated or unexpected layouts.
- Learning rate: GPU-resident linear warmup for 3% of the run, followed by cosine decay to 10% of the base LR.
- Checkpoints include model weights, optimizer moments, and the device optimizer step counter. New trainer saves also write a sidecar with dataset SHA-256 and run settings; resume refuses mismatches. Legacy checkpoints without a sidecar emit a warning.

## Sparse top-2 MoE mode

- JARVIS_MOE_SPARSE=1 compacts exactly the selected top-2 rows by expert and runs variable-size FP8 W1/GELU/W2 GEMMs; no routes are dropped or padded.
- The active path computes offsets on CUDA, copies expert counts to the host, and synchronizes once in each layer during forward and again during backward recomputation. Sparse mode disables CUDA Graph capture. The previous route-cache optimization was lost and is not claimed here.
- This restores the compact sparse execution path, but the host barriers may reduce throughput. Earlier benchmark figures above describe prior builds and are not valid for this revision; rerun matched quality and speed comparisons before treating sparse mode as faster.
- Cross-entropy reuses the logits allocation for gradients, avoiding a separate NT×Vpad BF16 buffer.
- cuBLASLt cached FP8 GEMMs fall back to BF16 when the heuristic returns no algorithm for a routed expert shape.
- Use JARVIS_MOE_SPARSE=0 to run the dense CUDA Graph path.

## Paper-fidelity mode

- Set `JARVIS_PAPER_FIDELITY=1` to enable the paper's Gaussian router perturbation (`sigma=0.1`) before softmax/top-2 selection. The noise is stateless, seeded by the device optimizer step, and graph-safe, so each update changes routing while forward and backward recomputation see the same perturbation.
- Paper mode is opt-in and can be paired with the default sparse route, or with dense mode by setting `JARVIS_MOE_SPARSE=0`.
- This is the first fidelity gate. The native engine still uses BF16 activations and FP8 cached GEMMs; a custom ternary ALU kernel is deferred until paper-mode loss/throughput are measured against the baseline.

## Paper and implementation decisions

- Keep AbsMean ternarization for production training. The literal paper quantizer collapses standard small-variance initialization toward all-zero projection weights, as documented by the ternary A/B experiments.
- FP8 is a forward-cache optimization, not the optimizer/master representation. Keeping optimizer state in FP32 protects ternary STE stability.
- `start.bat` is the supported Windows entry point for rebuild, training, resume, evaluation, benchmarking, and GPU diagnostics.

## Resume and validation

- Train: `python jarvis_engine/train_cuda.py`
- Resume: `python jarvis_engine/train_cuda.py resume`
- Agent Python and shell tools require typing `YES` at startup; without consent they stay disabled.
- Rebuild: `cd jarvis_engine/cuda_engine; set JARVIS_ALLOW_CUDA_MISMATCH=1; python build_engine.py`
- Validate code with `python -m py_compile ...` and `git diff --check`.\n