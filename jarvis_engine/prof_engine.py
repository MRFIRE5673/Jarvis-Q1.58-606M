import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jarvis_engine.cuda_engine import eng
eng.init(2, 512, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, 4, 1e-4, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0)
eng.warmup(1)
x = (np.arange(2 * 512 * 4) % 97 + 5).astype(np.int32)
print("PROF", eng.profile_step(x, 3))
