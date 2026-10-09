import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jarvis_engine.cuda_engine import eng
B, T, A = 2, 512, 4
lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
eng.init(B, T, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, A, lr, 0.9, 0.999, 1e-8, 0.0, 1.0, 0.01, 0.01, 0.1, 0.99, 0)
r = np.random.default_rng(0)
n = lambda *s: r.standard_normal(s, dtype=np.float32) * 0.02
o = lambda c: np.ones(c, np.float32)
eng.load_weight("emb", n(50304, 1024)); eng.load_weight("final_norm", o(1024))
for l in range(24):
    eng.load_weight(f"qkv.{l}", n(3072, 1024)); eng.load_weight(f"out.{l}", n(1024, 1024))
    eng.load_weight(f"w1.{l}", n(8192, 1024)); eng.load_weight(f"w2.{l}", n(4096, 2048))
    eng.load_weight(f"norm1.{l}", o(1024)); eng.load_weight(f"norm2.{l}", o(1024))
    eng.load_weight(f"gamma.{l}", np.full(16, np.log(0.95), np.float32))
    eng.load_weight(f"router.{l}", n(4, 1024))
x = (np.arange(B * T * A) % 97 + 5).astype(np.int32)
for i in range(12):
    print("LOSS", i, eng.train_step(x), flush=True)
