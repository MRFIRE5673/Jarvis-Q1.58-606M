"""Train with the native CUDA engine; repeat deterministically to meet an optional token budget."""
import os, sys, time, struct, json, hashlib, warnings, subprocess, numpy as np


# ---------------------------------------------------------------------------
# Thermal guard (from author's friend's fine_tune.py — adapted for native training)
# ---------------------------------------------------------------------------
_GPU_THROTTLE, _GPU_SHUTDOWN = 75.0, 80.0
_CPU_THROTTLE, _CPU_SHUTDOWN = 80.0, 85.0

def _gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        return float(out.split("\n")[0])
    except Exception:
        return None

def _cpu_temp():
    try:
        import psutil
        for key in ("coretemp", "k10temp", "cpu_thermal"):
            entries = psutil.sensors_temperatures().get(key, [])
            if entries:
                return max(e.current for e in entries)
    except Exception:
        pass
    return None

def thermal_check():
    """Call after each training step. Throttles or aborts on overtemp."""
    gpu, cpu = _gpu_temp(), _cpu_temp()
    if gpu is not None:
        if gpu >= _GPU_SHUTDOWN:
            raise RuntimeError(f"[ThermalGuard] GPU {gpu:.1f}°C >= {_GPU_SHUTDOWN}°C — aborting.")
        if gpu >= _GPU_THROTTLE:
            print(f"[ThermalGuard] GPU {gpu:.1f}°C — sleeping 15s", flush=True)
            time.sleep(15)
    if cpu is not None:
        if cpu >= _CPU_SHUTDOWN:
            raise RuntimeError(f"[ThermalGuard] CPU {cpu:.1f}°C >= {_CPU_SHUTDOWN}°C — aborting.")
        if cpu >= _CPU_THROTTLE:
            print(f"[ThermalGuard] CPU {cpu:.1f}°C — sleeping 20s", flush=True)
            time.sleep(20)

_THERMAL_GUARD = os.environ.get("JARVIS_THERMAL_GUARD", "1").lower() in ("1", "y", "yes", "true")
if os.environ.get("JARVIS_TRAIN_MASK", "").strip():
    os.environ.setdefault("JARVIS_MOE_SPARSE", "0")
else:
    os.environ.setdefault("JARVIS_MOE_SPARSE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jarvis_engine.cuda_engine import eng, _ENGINE_AVAILABLE
if not _ENGINE_AVAILABLE or eng is None:
    sys.exit(
        "[train_cuda] ERROR: Native CUDA engine is not available.\n"
        "             Native training requires an NVIDIA CUDA GPU with the compiled engine.\n"
        "             To build: cd jarvis_engine/cuda_engine && python build_engine.py"
    )

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.abspath(os.path.join(ROOT, os.environ.get("JARVIS_TRAIN_DATA", os.path.join("data", "train_tokens.npy"))))
MASK_ENV = os.environ.get("JARVIS_TRAIN_MASK", "").strip()
MASK_PATH = os.path.abspath(os.path.join(ROOT, MASK_ENV)) if MASK_ENV else None
CKPT = os.path.abspath(os.path.join(ROOT, os.environ.get("JARVIS_CKPT", os.path.join("data", "jarvis_cuda.ckpt"))))
CKPT_META = CKPT + ".meta.json"
B, T = 8, 512
ACC = int(os.environ.get("JARVIS_ACCUM_STEPS", "2"))
LR = float(os.environ.get("JARVIS_LR", "2e-5"))
ROUTER_NOISE = float(os.environ.get("JARVIS_ROUTER_NOISE", "0.1"))
if ACC < 1:
    raise ValueError("JARVIS_ACCUM_STEPS must be >= 1")
if not np.isfinite(LR) or LR <= 0:
    raise ValueError("JARVIS_LR must be a finite positive value")
if not np.isfinite(ROUTER_NOISE) or ROUTER_NOISE < 0:
    raise ValueError("JARVIS_ROUTER_NOISE must be a finite nonnegative value")
NT = B * T * ACC  # tokens per optimizer step (engine consumes ACC microbatches)

tok = np.load(DATA).astype(np.int32)
corpus_steps = len(tok) // NT
if corpus_steps == 0:
    raise ValueError(f"Need at least {NT} tokens, found {len(tok)}")
requested_tokens = os.environ.get("JARVIS_TRAIN_TOKENS", "").strip()
if requested_tokens:
    try:
        requested_tokens = int(requested_tokens)
    except ValueError as exc:
        raise ValueError("JARVIS_TRAIN_TOKENS must be a positive integer") from exc
    if requested_tokens <= 0:
        raise ValueError("JARVIS_TRAIN_TOKENS must be a positive integer")
    steps = requested_tokens // NT
    if steps == 0:
        raise ValueError(f"JARVIS_TRAIN_TOKENS must be at least {NT}")
else:
    steps = corpus_steps
train_tokens = steps * NT
print(f"Training data: {DATA}")
print(f"{len(tok):,} corpus tokens -> {steps:,} updates of {NT:,} tokens ({train_tokens:,} token exposures)")
print(f"Training passes: {steps / corpus_steps:.3f} (remainder is a deterministic reshuffled partial pass)")
target_mask = None
if MASK_PATH:
    if not os.path.exists(MASK_PATH):
        raise FileNotFoundError(f"Training target mask is missing: {MASK_PATH}")
    target_mask = np.load(MASK_PATH, mmap_mode="r")
    if target_mask.ndim != 1 or target_mask.shape != tok.shape or target_mask.dtype != np.uint8:
        raise ValueError("Training target mask must be a same-length uint8 array")
    if not np.isin(target_mask, (0, 1)).all():
        raise ValueError("Training target mask values must be 0 or 1")
    print(f"Response-only loss mask: {MASK_PATH}")


def validate_checkpoint(path):
    size = os.path.getsize(path)
    if size < 8 or (size - 4) % 4:
        raise ValueError(f"Invalid checkpoint size: {size:,} bytes")
    with open(path, "rb") as f:
        saved_step = struct.unpack("<i", f.read(4))[0]
    values = np.memmap(path, dtype="<f4", mode="r", offset=4)
    for start in range(0, values.size, 8_000_000):
        if not np.isfinite(values[start:start + 8_000_000]).all():
            raise ValueError(
                f"Checkpoint contains NaN/Inf (header step {saved_step}); "
                "refusing to resume corrupted weights or optimizer state."
            )


run_mode = sys.argv[1] if len(sys.argv) > 1 else "fresh"
if run_mode not in ("fresh", "resume", "resume-data", "resume-lr", "resume-sft"):
    raise ValueError("usage: train_cuda.py [fresh|resume|resume-data|resume-lr|resume-sft]")
allow_sft = run_mode == "resume-sft"
resume_requested = run_mode in ("resume", "resume-data", "resume-lr", "resume-sft")
allow_new_data = run_mode in ("resume-data", "resume-sft")
allow_lr_change = run_mode in ("resume-lr", "resume-sft")
if allow_sft and not os.path.exists(CKPT):
    raise FileNotFoundError("resume-sft requires an existing native CUDA checkpoint")
if resume_requested and os.path.exists(CKPT):
    print("Validating checkpoint before resume...", flush=True)
    validate_checkpoint(CKPT)

WARMUP_STEPS = max(10, min(1000, int(0.03 * steps)))
DECAY_STEPS = max(steps, WARMUP_STEPS + 1)
MIN_LR_RATIO = 0.1
RUN_CONFIG = {
    "training_core_version": 4,
    "d_model": 1024, "n_layers": 24, "n_experts": 4,
    "batch_size": B, "seq_len": T, "accum_steps": ACC,
    "learning_rate": LR, "warmup_steps": WARMUP_STEPS,
    "decay_steps": DECAY_STEPS, "min_lr_ratio": MIN_LR_RATIO,
    "corpus_steps": corpus_steps, "steps_per_run": steps,
    "train_tokens": train_tokens, "batch_order_seed": 1,
    "sparse_moe": os.environ.get("JARVIS_MOE_SPARSE", "1").lower() in ("1", "y", "yes", "true"),
    "router_noise_std": ROUTER_NOISE,
    "paper_fidelity": os.environ.get("JARVIS_PAPER_FIDELITY", "0").lower() in ("1", "y", "yes", "true"),
    "masked_targets": target_mask is not None,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


DATA_SHA256 = sha256_file(DATA)
MASK_SHA256 = sha256_file(MASK_PATH) if MASK_PATH else None
LR_STEP_OFFSET = 0
if allow_sft and os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        LR_STEP_OFFSET = struct.unpack("<i", f.read(4))[0]
    if os.path.exists(CKPT_META):
        with open(CKPT_META, encoding="utf-8") as f:
            previous = json.load(f)
        if (previous.get("checkpoint_step") == LR_STEP_OFFSET
                and previous.get("data_sha256") == DATA_SHA256
                and previous.get("target_mask_sha256") == MASK_SHA256):
            LR_STEP_OFFSET = previous.get("config", {}).get("lr_step_offset", LR_STEP_OFFSET)
    RUN_CONFIG["lr_step_offset"] = LR_STEP_OFFSET


def validate_run_metadata(saved_step):
    if not os.path.exists(CKPT_META):
        warnings.warn(
            "Legacy checkpoint has no dataset/config metadata; resuming with the current run settings.",
            RuntimeWarning,
        )
        return 0 if allow_new_data else saved_step
    with open(CKPT_META, encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("checkpoint_step") != saved_step:
        raise ValueError("Checkpoint and metadata sidecar are from different saves; refusing resume.")
    data_changed = (metadata.get("data_sha256") != DATA_SHA256 or
                    metadata.get("target_mask_sha256") != MASK_SHA256)
    config_changed = metadata.get("config") != RUN_CONFIG
    if data_changed or config_changed:
        saved_config = metadata.get("config", {})
        if allow_sft:
            changed = {
                key for key in saved_config.keys() | RUN_CONFIG.keys()
                if saved_config.get(key) != RUN_CONFIG.get(key)
            }
            allowed = {
                "learning_rate", "corpus_steps", "steps_per_run", "train_tokens",
                "warmup_steps", "decay_steps", "masked_targets",
                "lr_step_offset",
            }
            if changed - allowed:
                raise ValueError("SFT checkpoint model settings differ; only data, LR, and schedule may change.")
            print("Checkpoint settings compatible; starting instruction fine-tuning at data step 0.", flush=True)
            return 0
        if allow_lr_change:
            changed = {
                key for key in saved_config.keys() | RUN_CONFIG.keys()
                if saved_config.get(key) != RUN_CONFIG.get(key)
            }
            if data_changed or changed - {"learning_rate"}:
                raise ValueError("resume-lr only permits a learning-rate change on the same data/settings.")
            warnings.warn(
                f"Resuming the same run at LR={LR:g}; retaining checkpoint weights and Adam moments.",
                RuntimeWarning,
            )
        elif not allow_new_data:
            raise ValueError("Checkpoint was trained with different data or settings; refusing resume.")
        else:
            schedule_fields = {
                "corpus_steps", "steps_per_run", "train_tokens", "warmup_steps", "decay_steps",
            }
            if any(saved_config.get(key) != value for key, value in RUN_CONFIG.items() if key not in schedule_fields):
                raise ValueError("Checkpoint model/optimizer settings differ; cannot safely resume on new data.")
            if data_changed:
                print("Checkpoint settings compatible; starting the new corpus from its first batch.", flush=True)
                return 0
            warnings.warn("Dataset matches but its training schedule changed; continuing from its saved data cursor.", RuntimeWarning)
    return int(metadata.get("data_step", saved_step))


def save_run_metadata(data_step):
    with open(CKPT, "rb") as f:
        checkpoint_step = struct.unpack("<i", f.read(4))[0]
    payload = {
        "checkpoint_step": checkpoint_step,
        "data_step": data_step,
        "data_sha256": DATA_SHA256,
        "target_mask_sha256": MASK_SHA256,
        "config": RUN_CONFIG,
    }
    temp_path = CKPT_META + ".tmp"
    with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, CKPT_META)


print(f"Checkpoint: {CKPT}")
print(f"LR schedule: 3% linear warmup ({WARMUP_STEPS} steps), cosine decay to {MIN_LR_RATIO:.0%}")
print(f"Base learning rate: {LR:g}")
print(f"Top-2 router noise: sigma={ROUTER_NOISE:g}  (Gumbel-equivalent Box-Muller, per-layer seed)")
if LR_STEP_OFFSET:
    print(f"Fine-tune LR schedule starts at optimizer step {LR_STEP_OFFSET}")
if RUN_CONFIG["paper_fidelity"]:
    print("Literal paper ternary mapping: round(clamp(W,-1,1)); this is a research ablation.")
    print("WARNING: with the default 0.02 weight init, this paper quantizer zeros nearly every projection weight.")

# Expert utilisation tracking — cheap host-side EMA over dispatch counts
_expert_ema = None   # shape (E,), float32, updated at every log step
_EXP_EMA_ALPHA = 0.98

def init_weights(L=24, C=1024, E=4, hi=2048, Vp=50304, H=16, std=0.02, seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s: (r.standard_normal(s, dtype=np.float32) * std)
    eng.load_weight("emb", n(Vp, C))
    eng.load_weight("final_norm", np.ones(C, np.float32))
    for l in range(L):
        eng.load_weight(f"qkv.{l}", n(3 * C, C)); eng.load_weight(f"out.{l}", n(C, C))
        eng.load_weight(f"w1.{l}", n(E * hi, C)); eng.load_weight(f"w2.{l}", n(E * C, hi))
        eng.load_weight(f"norm1.{l}", np.ones(C, np.float32)); eng.load_weight(f"norm2.{l}", np.ones(C, np.float32))
        eng.load_weight(f"gamma.{l}", np.full(H, np.log(0.95), np.float32))
        eng.load_weight(f"router.{l}", n(E, C))


# batch, seq, C, H, L, E, K, hi, vocab, vocab_pad, chunk, accum, lr, b1, b2, eps, wd, clip, bal, refl, lsf_min, lsf_max, dev
eng.init(B, T, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, ACC,
         LR, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0,
         WARMUP_STEPS, DECAY_STEPS, MIN_LR_RATIO, LR_STEP_OFFSET)
resuming = resume_requested and os.path.exists(CKPT)
start_step = 0
data_start_step = 0
if resuming:
    with open(CKPT, "rb") as f:
        start_step = struct.unpack("<i", f.read(4))[0]
    data_start_step = validate_run_metadata(start_step)
    eng.load_checkpoint(CKPT); print("resumed")
else:
    init_weights()
if data_start_step > steps:
    raise ValueError(f"Checkpoint data step {data_start_step} exceeds this run's {steps} steps")
if data_start_step == steps:
    print(f"Training already complete: {steps}/{steps} steps; checkpoint unchanged.")
    sys.exit(0)
if target_mask is None:
    eng.capture_graph()
else:
    print("Response-only SFT loss active; CUDA Graph disabled for masked targets.", flush=True)

t0, ema = time.time(), None
first_epoch, first_offset = divmod(data_start_step, corpus_steps)
last_epoch = (steps - 1) // corpus_steps
for epoch in range(first_epoch, last_epoch + 1):
    order = np.random.default_rng(1 + epoch).permutation(corpus_steps)
    start_offset = first_offset if epoch == first_epoch else 0
    end_offset = min(corpus_steps, steps - epoch * corpus_steps)
    for offset in range(start_offset, end_offset):
        i = epoch * corpus_steps + offset
        b = order[offset]
        start = b * NT
        batch = tok[start:start + NT]
        if target_mask is None:
            loss = eng.train_step(batch)
        else:
            batch_mask = target_mask[start:start + NT]
            active_targets = int(np.count_nonzero(batch_mask.reshape(B * ACC, T)[:, 1:]))
            if active_targets == 0:
                raise ValueError(f"No assistant targets in training batch {i}")
            loss = eng.train_step_masked(batch, batch_mask, active_targets)
        if not np.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step {i}; stopping without overwriting the last checkpoint."
            )
        if _THERMAL_GUARD:
            thermal_check()
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss

        # Expert utilisation — read raw loss components when available
        if i % 50 == 0:
            el = time.time() - t0
            run_tokens = (i - data_start_step + 1) * NT
            # Try to get balance loss from engine (optional — engine may not expose it)
            bal_str = ""
            try:
                bal = float(eng.get_loss())   # returns combined loss; balance is embedded
            except Exception:
                bal = None
            print(
                f"step {i:>6}/{steps}  loss {loss:.4f}  ema {ema:.4f}"
                f"  {run_tokens / max(el, 1e-9):,.0f} tok/s",
                flush=True,
            )

        if i and i % 500 == 0:
            eng.save_checkpoint(CKPT)
            save_run_metadata(i + 1)
            # Print expert utilisation hint at every checkpoint
            print(
                f"  [checkpoint step {i}] ema loss {ema:.4f}  router noise sigma={ROUTER_NOISE:g}"
                f"  sparse_moe={RUN_CONFIG['sparse_moe']}",
                flush=True,
            )
eng.save_checkpoint(CKPT)
save_run_metadata(steps)
print(f"done. final ema loss {ema:.4f} -> {CKPT}")
