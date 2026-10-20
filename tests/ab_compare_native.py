"""Short controlled native CUDA A/B run from one checkpoint."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jarvis_engine.cuda_engine import eng

ACC = 2
B, T = 4, 512
NT = B * T * ACC
STEPS = 3
base = os.path.join(os.path.dirname(__file__), "..", "data", "jarvis_cuda.ckpt")
mode = os.environ.get("AB_MODE", "baseline")

eng.init(B, T, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, ACC,
         3e-4, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0,
         10, 100, 0.1)
eng.load_checkpoint(base)
tokens = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "train_tokens.npy"))[:NT * STEPS].astype(np.int32)
if mode == "baseline":
    eng.capture_graph()
else:
    eng.warmup(1)

losses = []
t0 = time.perf_counter()
for i in range(STEPS):
    losses.append(float(eng.train_step(tokens[i * NT:(i + 1) * NT])))
elapsed = time.perf_counter() - t0
print(f"MODE={mode} LOSSES=" + ",".join(f"{x:.6f}" for x in losses))
print(f"MODE={mode} TOK_S={(STEPS * NT) / elapsed:.1f} WALL_S={elapsed:.3f}")
eng.save_checkpoint(os.path.join(os.path.dirname(__file__), "..", "data", f"ab_{mode}.ckpt"))
eng.destroy()
