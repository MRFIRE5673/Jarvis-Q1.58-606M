"""Build data/train_tokens.npy: ~N tokens, 50% English (fineweb-edu) / 50% code (codeparrot). GPT-2 BPE."""
import os, sys, numpy as np, tiktoken
from datasets import load_dataset

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "train_tokens.npy")
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def collect(name, kw, field, n):
    ds = load_dataset(name, split="train", streaming=True, **kw)
    out, got = [], 0
    for ex in ds:
        t = enc.encode_ordinary(ex[field]) + [EOT]
        out.append(np.array(t, dtype=np.uint16)); got += len(t)
        if got % 500_000 < len(t): print(f"  {name}: {got:,}/{n:,}", flush=True)
        if got >= n: break
    return np.concatenate(out)[:n]


half = TOTAL // 2
eng = collect("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, "text", half)
code = collect("codeparrot/codeparrot-clean", {}, "content", half)
# interleave in 512-token blocks so every step sees both domains
B = 512
def blocks(a): return a[: len(a) // B * B].reshape(-1, B)
e, c = blocks(eng), blocks(code)
n = min(len(e), len(c))
mix = np.stack([e[:n], c[:n]], 1).reshape(-1, B)
rng = np.random.default_rng(0); rng.shuffle(mix)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.save(OUT, mix.reshape(-1))
print(f"saved {mix.size:,} tokens ({n*B:,} English + {n*B:,} code) -> {OUT}")
