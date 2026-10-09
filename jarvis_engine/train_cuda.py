"""Train on data/train_tokens.npy with the native CUDA engine (1 pass over the tokens)."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jarvis_engine.cuda_engine import eng

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "train_tokens.npy")
CKPT = os.path.join(ROOT, "data", "jarvis_cuda.ckpt")
B, T, ACC = 2, 512, 4
NT = B * T * ACC  # tokens per optimizer step (engine consumes ACC microbatches)

tok = np.load(DATA).astype(np.int32)
steps = len(tok) // NT
print(f"{len(tok):,} tokens -> {steps} steps of {NT} tokens")

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
         3e-4, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0)
if len(sys.argv) > 1 and sys.argv[1] == "resume" and os.path.exists(CKPT):
    eng.load_checkpoint(CKPT); print("resumed")
else:
    init_weights()
eng.warmup(3)
eng.capture_graph()

order = np.random.default_rng(1).permutation(steps)
t0, ema = time.time(), None
for i, b in enumerate(order):
    loss = eng.train_step(tok[b * NT:(b + 1) * NT])
    ema = loss if ema is None else 0.98 * ema + 0.02 * loss
    if i % 50 == 0:
        el = time.time() - t0
        print(f"step {i}/{steps} loss {loss:.4f} ema {ema:.4f} {(i+1)*NT/max(el,1e-9):.0f} tok/s", flush=True)
    if i and i % 500 == 0:
        eng.save_checkpoint(CKPT)
eng.save_checkpoint(CKPT)
print(f"done. final ema loss {ema:.4f} -> {CKPT}")
