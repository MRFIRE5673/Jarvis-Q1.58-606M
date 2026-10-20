"""Train on data/train_tokens.npy with the native CUDA engine (1 pass over the tokens)."""
import os, sys, time, struct, json, hashlib, warnings, numpy as np
os.environ.setdefault("JARVIS_MOE_SPARSE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jarvis_engine.cuda_engine import eng

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.abspath(os.path.join(ROOT, os.environ.get("JARVIS_TRAIN_DATA", os.path.join("data", "train_tokens.npy"))))
CKPT = os.path.join(ROOT, "data", "jarvis_cuda.ckpt")
CKPT_META = CKPT + ".meta.json"
B, T = 8, 512
ACC = int(os.environ.get("JARVIS_ACCUM_STEPS", "2"))
LR = float(os.environ.get("JARVIS_LR", "1e-4"))
if ACC < 1:
    raise ValueError("JARVIS_ACCUM_STEPS must be >= 1")
if not np.isfinite(LR) or LR <= 0:
    raise ValueError("JARVIS_LR must be a finite positive value")
NT = B * T * ACC  # tokens per optimizer step (engine consumes ACC microbatches)

tok = np.load(DATA).astype(np.int32)
steps = len(tok) // NT
if steps == 0:
    raise ValueError(f"Need at least {NT} tokens, found {len(tok)}")
print(f"Training data: {DATA}")
print(f"{len(tok):,} tokens -> {steps} steps of {NT} tokens")


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
if run_mode not in ("fresh", "resume", "resume-data"):
    raise ValueError("usage: train_cuda.py [fresh|resume|resume-data]")
resume_requested = run_mode in ("resume", "resume-data")
allow_new_data = run_mode == "resume-data"
if resume_requested and os.path.exists(CKPT):
    print("Validating checkpoint before resume...", flush=True)
    validate_checkpoint(CKPT)

WARMUP_STEPS = max(10, min(1000, int(0.03 * steps)))
DECAY_STEPS = max(steps, WARMUP_STEPS + 1)
MIN_LR_RATIO = 0.1
RUN_CONFIG = {
    "d_model": 1024, "n_layers": 24, "n_experts": 4,
    "batch_size": B, "seq_len": T, "accum_steps": ACC,
    "learning_rate": LR, "warmup_steps": WARMUP_STEPS,
    "decay_steps": DECAY_STEPS, "min_lr_ratio": MIN_LR_RATIO,
    "steps_per_epoch": steps, "batch_order_seed": 1,
    "sparse_moe": os.environ.get("JARVIS_MOE_SPARSE", "1").lower() in ("1", "y", "yes", "true"),
    "paper_fidelity": os.environ.get("JARVIS_PAPER_FIDELITY", "0").lower() in ("1", "y", "yes", "true"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


DATA_SHA256 = sha256_file(DATA)


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
    data_changed = metadata.get("data_sha256") != DATA_SHA256
    config_changed = metadata.get("config") != RUN_CONFIG
    if data_changed or config_changed:
        if not allow_new_data:
            raise ValueError("Checkpoint was trained with different data or settings; refusing resume.")
        schedule_fields = {"steps_per_epoch", "warmup_steps", "decay_steps"}
        saved_config = metadata.get("config", {})
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
        "config": RUN_CONFIG,
    }
    temp_path = CKPT_META + ".tmp"
    with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, CKPT_META)


print(f"LR schedule: 3% linear warmup ({WARMUP_STEPS} steps), cosine decay to {MIN_LR_RATIO:.0%}")
print(f"Base learning rate: {LR:g}")

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
         WARMUP_STEPS, DECAY_STEPS, MIN_LR_RATIO)
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
eng.capture_graph()

order = np.random.default_rng(1).permutation(steps)[data_start_step:]
t0, ema = time.time(), None
for i, b in enumerate(order, start=data_start_step):
    loss = eng.train_step(tok[b * NT:(b + 1) * NT])
    if not np.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss at step {i}; stopping without overwriting the last checkpoint."
        )
    ema = loss if ema is None else 0.98 * ema + 0.02 * loss
    if i % 50 == 0:
        el = time.time() - t0
        run_tokens = (i - data_start_step + 1) * NT
        print(f"step {i}/{steps} loss {loss:.4f} ema {ema:.4f} {run_tokens/max(el,1e-9):.0f} tok/s", flush=True)
    if i and i % 500 == 0:
        eng.save_checkpoint(CKPT)
        save_run_metadata(i + 1)
eng.save_checkpoint(CKPT)
save_run_metadata(steps)
print(f"done. final ema loss {ema:.4f} -> {CKPT}")
