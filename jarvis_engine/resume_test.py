"""Load data/jarvis_cuda.ckpt and print loss on a few batches (expect ~2, not ~8.5)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jarvis_engine.cuda_engine import eng
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
B, T, ACC = 2, 512, 4; NT = B * T * ACC
tok = np.load(os.path.join(R, "train_tokens.npy")).astype(np.int32)
eng.init(B, T, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, ACC,
         1e-8, 0.9, 0.999, 1e-8, 0.0, 1.0, 0.01, 0.01, 0.1, 0.99, 0)
eng.load_checkpoint(os.path.join(R, "jarvis_cuda.ckpt"))
eng.warmup(1); eng.capture_graph()
for i in range(5):
    print("RLOSS", eng.train_step(tok[i * NT:(i + 1) * NT]))
