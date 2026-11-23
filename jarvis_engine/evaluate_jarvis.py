# evaluate_jarvis.py
"""
Jarvis 606M Production Evaluation & Interactive Inference Suite
===============================================================
Features:
  1. Interactive terminal CLI (`--interactive`) with streaming token output
  2. Direct prompt evaluation (`--prompt "..."`)
  3. Strict 606M checkpoint loading with full backend inspection
  4. Generation controls: --max_new_tokens, --temperature, --top_k, --top_p
  5. State isolation (`model.reset_state()` per prompt) and slash commands (/reset, /info, /quit)
  6. Quantitative holdout corpus evaluation & perplexity tracking
"""

import os
import sys
import math
import time
import argparse
import statistics
import contextlib
import torch
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure repo paths are in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JARVIS_ENGINE_PATH = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
SPARSE_DIR = os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")
ATTN_DIR = os.path.join(WORKSPACE_ROOT, "associative_attention_cuda")

for p in [WORKSPACE_ROOT, JARVIS_ENGINE_PATH, SPARSE_DIR, ATTN_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import Jarvis
from tokenizer import ENCODING, StreamingDecoder


def resolve_device(device_arg="auto"):
    """Resolves target execution device (cuda, cpu, mps, or auto-detect)."""
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_name(device):
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def get_autocast(device):
    if device == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    elif device == "cpu":
        try:
            return torch.amp.autocast('cpu', dtype=torch.bfloat16)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def sync_device(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_device_cache(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_path(path_arg, default_name=None):
    """Resolves relative file paths whether running from root, data/, or jarvis_engine/."""
    if path_arg and os.path.exists(path_arg):
        return os.path.abspath(path_arg)
    candidates = []
    if path_arg:
        candidates.append(path_arg)
        candidates.append(os.path.join(JARVIS_ENGINE_PATH, path_arg))
        candidates.append(os.path.join(WORKSPACE_ROOT, path_arg))
        candidates.append(os.path.join(WORKSPACE_ROOT, "data", path_arg))
    if default_name:
        candidates.append(default_name)
        candidates.append(os.path.join(JARVIS_ENGINE_PATH, default_name))
        candidates.append(os.path.join(WORKSPACE_ROOT, default_name))
        candidates.append(os.path.join(WORKSPACE_ROOT, "data", default_name))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return path_arg or default_name


def parse_args():
    parser = argparse.ArgumentParser(description="Jarvis 606M Production Evaluation and Interactive Suite")
    # Interactive / Generation controls
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Launch interactive conversation terminal")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to evaluate directly")
    parser.add_argument("--raw", action="store_true",
                        help="Disable instruction framing (User: ... Assistant:) and generate raw text continuation")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate per response (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0.0 for greedy decoding, default: 0.7)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering threshold (default: 50, <=0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus top-p filtering threshold (default: 0.9, 1.0 to disable)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"],
                        help="Execution device: auto, cuda, cpu, or mps (default: auto)")

    # Quantitative corpus evaluation arguments
    parser.add_argument("--eval-file", type=str, default="fresh_holdout.txt",
                        help="Path to genuine holdout text file (must NOT be data.txt)")
    parser.add_argument("--train-file", type=str, default="data.txt",
                        help="Path to training corpus file for comparison")
    parser.add_argument("--num-windows", type=int, default=200,
                        help="Number of evaluation windows (default: 200)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length of each evaluation window (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic window selection")
    parser.add_argument("--skip-train-eval", action="store_true",
                        help="Skip training corpus evaluation and only run holdout")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip text generation evaluation")
    return parser.parse_args()


def load_binary_ckpt(resolved_ckpt, model, device="cuda"):
    import struct, numpy as np

    def native_quantize(weight):
        # Native training shares one AbsMean scale across each stored matrix.
        weight = weight.float()
        alpha = weight.abs().mean()
        return (weight.div(alpha + 1e-8).clamp(-1, 1).round() * alpha).to(torch.bfloat16).float()

    with open(resolved_ckpt, "rb") as f:
        step = struct.unpack("i", f.read(4))[0]
        Vp, C, L, E, hi, H = 50304, 1024, 24, 4, 2048, 16
        def read_tensor(shape):
            n = 1
            for s in shape: n *= s
            data = np.frombuffer(f.read(n * 4), dtype=np.float32).copy().reshape(shape)
            return torch.from_numpy(data)

        tok_emb = read_tensor((Vp, C))[:50257]
        final_norm = read_tensor((C,))

        model.tok_emb.weight.data.copy_(tok_emb.to(device=device, dtype=model.tok_emb.weight.dtype))
        model.lm_head.weight.data.copy_(tok_emb.to(device=device, dtype=model.lm_head.weight.dtype))
        model.final_norm.weight.data.copy_(final_norm.to(device=device, dtype=model.final_norm.weight.dtype))

        for l in range(L):
            qkv = native_quantize(read_tensor((3 * C, C)))
            out = native_quantize(read_tensor((C, C)))
            # The native engine quantizes all experts in a layer with one scale.
            w1 = native_quantize(read_tensor((E * hi, C)).view(E, hi, C))
            w2 = native_quantize(read_tensor((E * C, hi)).view(E, C, hi))
            norm1 = read_tensor((C,))
            norm2 = read_tensor((C,))
            gamma = read_tensor((H,))

            b = model.blocks[l]
            b.norm1.weight.data.copy_(norm1.to(device=device, dtype=b.norm1.weight.dtype))
            b.norm2.weight.data.copy_(norm2.to(device=device, dtype=b.norm2.weight.dtype))
            # Native engine stores raw unconstrained logits directly, matching PyTorch gamma_raw
            b.attn.gamma_raw.data.copy_(gamma.to(device=device, dtype=b.attn.gamma_raw.dtype))
            if hasattr(b.attn, "q_proj"):
                b.attn.q_proj.weight.data.copy_(qkv[:C].to(device=device, dtype=b.attn.q_proj.weight.dtype))
                b.attn.k_proj.weight.data.copy_(qkv[C:2*C].to(device=device, dtype=b.attn.k_proj.weight.dtype))
                b.attn.v_proj.weight.data.copy_(qkv[2*C:].to(device=device, dtype=b.attn.v_proj.weight.dtype))
                for proj in (b.attn.q_proj, b.attn.k_proj, b.attn.v_proj):
                    proj._native_prequantized = True
            elif hasattr(b.attn, "qkv_proj"):
                b.attn.qkv_proj.weight.data.copy_(qkv.to(device=device, dtype=b.attn.qkv_proj.weight.dtype))
                b.attn.qkv_proj._native_prequantized = True
            b.attn.out_proj.weight.data.copy_(out.to(device=device, dtype=b.attn.out_proj.weight.dtype))
            b.attn.out_proj._native_prequantized = True

            for e in range(E):
                b.moe.w1[e].weight.data.copy_(w1[e].to(device=device, dtype=b.moe.w1[e].weight.dtype))
                b.moe.w2[e].weight.data.copy_(w2[e].to(device=device, dtype=b.moe.w2[e].weight.dtype))
                b.moe.w1[e]._native_prequantized = True
                b.moe.w2[e]._native_prequantized = True
            b.moe._native_prequantized = True

        rem = f.read(L * E * C * 4)
        if len(rem) >= L * E * C * 4:
            r_data = np.frombuffer(rem, dtype=np.float32).copy().reshape((L, E, C))
            for l in range(L):
                model.blocks[l].moe.router.weight.data.copy_(torch.from_numpy(r_data[l]).to(device=device, dtype=model.blocks[l].moe.router.weight.dtype))
    return step


def load_jarvis_model(ckpt_path=None, device="auto"):
    device = resolve_device(device)
    resolved_ckpt = None
    if ckpt_path and os.path.exists(ckpt_path):
        resolved_ckpt = os.path.abspath(ckpt_path)
    else:
        for candidate in [ckpt_path, "ckpt_step_0004209.pt", "data/jarvis_cuda.ckpt", "data/ckpt_step_0004209.pt"]:
            if candidate:
                c = resolve_path(candidate)
                if c and os.path.exists(c):
                    resolved_ckpt = c
                    break

    print("=" * 80)
    print("                    JARVIS 606M MODEL INITIALIZATION                    ")
    print("=" * 80)
    model = Jarvis(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        num_experts=4,
        top_k=2,
        max_seq_len=256,
        use_cuda_attn=(device == "cuda"),
        use_cuda_moe=(device == "cuda")
    ).to(device)

    n_params, param_str = model.param_count()
    print(f"Model Architecture: 606M Jarvis ({n_params:,} parameters, {param_str})")

    if resolved_ckpt and os.path.exists(resolved_ckpt):
        print(f"Loading checkpoint from: {resolved_ckpt}...")
        if resolved_ckpt.endswith(".ckpt"):
            trained_step = load_binary_ckpt(resolved_ckpt, model, device=device)
            print(f"Native binary checkpoint loaded successfully (step: {trained_step})")
        else:
            # State-dict checkpoints do not need pickle execution. Keep loading
            # restricted so a downloaded or user-supplied checkpoint cannot run code.
            ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=True)
            state_dict = ckpt["model_state_dict"]
            new_state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                              for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
            assert len(missing) == 0 and len(unexpected) == 0, f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            trained_step = ckpt.get("step", "unknown")
            print(f"PyTorch checkpoint loaded successfully: strict=True verified (step: {trained_step})")
            del ckpt, state_dict, new_state_dict
    else:
        trained_step = 0
        resolved_ckpt = "None (untrained / fresh weights)"
        print("[Notice] No checkpoint found. Model initialized with random weights.")
        print("         To train the model on tokens, use Train / resume in start.bat / start.sh.")

    status = model.get_backend_status()
    device_name = get_device_name(device)
    print(f"Attention Backend:  {status['attn_backend'].upper()} ({status['attn_cuda_blocks']} blocks)")
    print(f"MoE Backend:        {status['moe_backend'].upper()} ({status['moe_cuda_blocks']} blocks)")
    print(f"Inference Device:   {device} ({device_name})")
    print(f"Inference Dtype:    torch.bfloat16 (with native FP32 accumulations)")
    print("=" * 80 + "\n")

    empty_device_cache(device)
    model.eval()
    return model, resolved_ckpt, trained_step


def print_model_info(model, ckpt_path, device="auto"):
    device = resolve_device(device)
    status = model.get_backend_status()
    n_params, param_str = model.param_count()
    device_name = get_device_name(device)
    vram_alloc = (torch.cuda.memory_allocated() / (1024 ** 3)) if (device == "cuda" and torch.cuda.is_available()) else 0.0
    vram_res = (torch.cuda.memory_reserved() / (1024 ** 3)) if (device == "cuda" and torch.cuda.is_available()) else 0.0

    print("\n--- Model & System Information ---")
    print(f"Checkpoint:          {ckpt_path}")
    print(f"Parameter Count:     {n_params:,} ({param_str})")
    print(f"Attention Backend:   {status['attn_backend'].upper()} ({status['attn_cuda_blocks']} blocks)")
    print(f"Sparse MoE Backend:  {status['moe_backend'].upper()} ({status['moe_cuda_blocks']} blocks)")
    print(f"Device:              {device} ({device_name})")
    print(f"Current VRAM Usage:  {vram_alloc:.2f} GB allocated / {vram_res:.2f} GB reserved")
    print(f"Liquid State Status: {sum(1 for h in model._h_states if h is not None)}/{len(model.blocks)} blocks active")
    print(f"Token Position:      {model._token_pos}")
    print("----------------------------------\n")


def sample_next_token(
    logits,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.15,
    prev_tokens=None,
    no_repeat_ngram_size=3
):
    """
    Samples next token with advanced anti-gibberish / anti-looping protections:
    1. N-gram repetition blocking (eliminates repeating 3-grams)
    2. Frequency-weighted exponential repetition penalty
    3. Consecutive single-token stutter suppression
    4. Temperature scaling
    5. Top-k and top-p nucleus sampling
    """
    logits = logits.clone().float()

    if prev_tokens is not None and len(prev_tokens) > 0:
        # 1. N-gram repetition blocking
        if no_repeat_ngram_size > 0 and len(prev_tokens) >= no_repeat_ngram_size:
            ngram_prefix = tuple(prev_tokens[-(no_repeat_ngram_size - 1):])
            banned_tokens = set()
            for idx in range(len(prev_tokens) - no_repeat_ngram_size + 1):
                if tuple(prev_tokens[idx : idx + no_repeat_ngram_size - 1]) == ngram_prefix:
                    banned_tokens.add(prev_tokens[idx + no_repeat_ngram_size - 1])
            if banned_tokens:
                banned_idx = [t for t in banned_tokens if 0 <= t < logits.size(-1)]
                if banned_idx:
                    logits[..., banned_idx] = -float("inf")

        # 2. Frequency-weighted repetition penalty
        if repetition_penalty > 1.0:
            from collections import Counter
            recent_counts = Counter(prev_tokens[-256:])
            for tok, count in recent_counts.items():
                if 0 <= tok < logits.size(-1):
                    penalty = repetition_penalty ** min(count, 4)
                    val = logits[..., tok]
                    logits[..., tok] = torch.where(val > 0, val / penalty, val * penalty)

            # 3. Consecutive repeat dampening (prevents whitespace/token stutter loops)
            last_tok = prev_tokens[-1]
            if 0 <= last_tok < logits.size(-1):
                logits[..., last_tok] -= 2.0

    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-5)
    if top_k is not None and top_k > 0:
        candidate_logits, candidate_ids = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1, sorted=True)
    else:
        candidate_logits, candidate_ids = torch.sort(logits, descending=True, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        cumulative_probs = torch.cumsum(F.softmax(candidate_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        candidate_logits = candidate_logits.masked_fill(remove, -float("inf"))

    probs = torch.nan_to_num(F.softmax(candidate_logits, dim=-1), nan=0.0)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    fallback = F.one_hot(candidate_logits.argmax(dim=-1), num_classes=candidate_logits.size(-1)).to(probs.dtype)
    probs = torch.where(probs_sum > 0, probs / probs_sum.clamp_min(1e-20), fallback)
    selected = torch.multinomial(probs, num_samples=1)
    return candidate_ids.gather(-1, selected)


_NATIVE_ENGINE = None

def get_native_engine(ckpt_path=None):
    """Initializes and returns the singleton native CUDA C++ inference engine."""
    global _NATIVE_ENGINE
    if _NATIVE_ENGINE is not None:
        return _NATIVE_ENGINE
    try:
        from jarvis_engine.cuda_engine import eng
        eng.init(
            1, 512, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, 1,
            1e-4, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0
        )
        resolved = ckpt_path
        if not resolved or not os.path.exists(resolved):
            resolved = resolve_path("data/jarvis_cuda.ckpt")
        if resolved and os.path.exists(resolved) and resolved.endswith(".ckpt"):
            eng.load_checkpoint(resolved)
        _NATIVE_ENGINE = eng
        return _NATIVE_ENGINE
    except Exception as e:
        return None


@torch.inference_mode()
def generate_streaming(
    model,
    enc,
    prompt_text: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "auto",
    ckpt_path: str = None
):
    """
    O(1) per-token streaming inference.
    Fast path: Native C++/CUDA Engine (100+ tok/s).
    Fallback path: PyTorch associative state caching.
    """
    device = resolve_device(device)
    # 1. Try ultra-fast Native C++/CUDA Engine
    if device == "cuda" and torch.cuda.is_available():
        native_eng = get_native_engine(ckpt_path)
        if native_eng is not None:
            prompt_ids = enc.encode_ordinary(prompt_text)
            if not prompt_ids:
                prompt_ids = [enc.eot_token]

            decoder = StreamingDecoder(enc)
            decoded_parts = []

            def stream_cb(tok_id):
                chunk = decoder.push(tok_id)
                decoded_parts.append(chunk)
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()

            sync_device(device)
            t_start = time.perf_counter()
            tokens = native_eng.generate(prompt_ids, max_new_tokens, temperature, top_p, top_k, stream_cb)
            sync_device(device)
            t_end = time.perf_counter()

            tail = decoder.finish()
            decoded_parts.append(tail)
            sys.stdout.write(tail + "\n")
            sys.stdout.flush()

            duration = t_end - t_start
            n_tokens = len(tokens)
            tok_s = n_tokens / max(duration, 1e-5)
            peak_vram = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

            print("-" * 50)
            print(f"Tokens Generated:  {n_tokens}")
            print(f"Generation Time:   {duration:.3f} s")
            print(f"Throughput:        {tok_s:.1f} tok/s  [Native C++/CUDA Engine]")

            return "".join(decoded_parts), {
                "tokens_generated": n_tokens,
                "generation_time_s": duration,
                "tokens_per_sec": tok_s,
                "peak_vram_gb": peak_vram,
            }

    # 2. Fallback PyTorch Path
    model.eval()
    model.reset_state()

    prompt_ids = enc.encode_ordinary(prompt_text)
    if len(prompt_ids) == 0:
        prompt_ids = [enc.eot_token]

    # --- Prefill: process full prompt in one shot, cache recurrent state ---
    context = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    with get_autocast(device):
        logits, _ = model(context, persist_state=True)
    # seed: last token of prompt gives first generation logit
    next_logits = logits[:, -1, :].float()
    next_tok = sample_next_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                 repetition_penalty=repetition_penalty, prev_tokens=prompt_ids)

    generated_token_ids = []
    decoder = StreamingDecoder(enc)
    decoded_parts = []

    sync_device(device)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    for step_idx in range(max_new_tokens):
        tok_id = next_tok.item()
        if tok_id == enc.eot_token:
            if step_idx == 0:
                print("[Note: Model immediately emitted EOT — try a different prompt or check checkpoint.]", flush=True)
            break

        generated_token_ids.append(tok_id)

        # Decode only this token's bytes; the incremental decoder handles split UTF-8 characters.
        new_text = decoder.push(tok_id)
        decoded_parts.append(new_text)
        if new_text:
            sys.stdout.write(new_text)
            sys.stdout.flush()

        # O(1) single-token forward: feeds only the new token, S_t and H_t carried
        with get_autocast(device):
            logits, _ = model(next_tok, persist_state=True)
        next_logits = logits[:, -1, :].float()
        seen = prompt_ids + generated_token_ids
        next_tok = sample_next_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                     repetition_penalty=repetition_penalty, prev_tokens=seen)

    sync_device(device)
    t_end = time.perf_counter()

    tail = decoder.finish()
    decoded_parts.append(tail)
    sys.stdout.write(tail + "\n")
    sys.stdout.flush()

    duration = t_end - t_start
    n_tokens = len(generated_token_ids)
    tok_s = n_tokens / max(duration, 1e-5)
    peak_vram = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if (device == "cuda" and torch.cuda.is_available()) else 0.0

    stats = {
        "tokens_generated": n_tokens,
        "generation_time_s": duration,
        "tokens_per_sec": tok_s,
        "peak_vram_gb": peak_vram,
    }

    print("-" * 50)
    print(f"Tokens Generated:  {n_tokens}")
    print(f"Generation Time:   {duration:.3f} s")
    print(f"Throughput:        {tok_s:.1f} tok/s  [Phase 5: O(1) state cache]")
    if device == "cuda" and torch.cuda.is_available():
        print(f"Peak VRAM:         {peak_vram:.2f} GB")
    print("-" * 50)

    # Clean model state after response completion
    model.reset_state()
    return "".join(decoded_parts), stats


def interactive_terminal(model, enc, args, ckpt_path, device="auto"):
    """Interactive CMD/Terminal REPL loop."""
    device = resolve_device(device)
    print("================================================================================")
    print("                   JARVIS 606M INTERACTIVE TERMINAL                             ")
    print("================================================================================")
    print("Commands:")
    print("  /reset  - Clear model liquid state and position offsets")
    print("  /info   - Display current model architecture, backends, and memory usage")
    print("  /quit   - Exit the interactive terminal (or /exit)")
    print(f"Controls: max_new_tokens={args.max_new_tokens}, temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("Type your prompt and press Enter to stream completions in real-time.")
    print("================================================================================\n")

    while True:
        try:
            prompt = input("Jarvis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive terminal.")
            break

        if not prompt:
            continue

        cmd = prompt.lower()
        if cmd in ["/quit", "/exit"]:
            print("Exiting Jarvis terminal. Goodbye!")
            break
        elif cmd == "/reset":
            model.reset_state()
            print("  [State] Model liquid membrane states and positional offsets cleared.")
            continue
        elif cmd == "/info":
            print_model_info(model, ckpt_path, device)
            continue

        prompt_text = prompt
        if not getattr(args, "raw", False) and not (prompt.startswith("User:") or prompt.startswith("System:")):
            prompt_text = f"User: {prompt}\nAssistant: "

        print("\nResponse:")
        generate_streaming(
            model=model,
            enc=enc,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            ckpt_path=ckpt_path
        )


@torch.no_grad()
def evaluate_corpus(model, file_path, enc, num_windows=200, seq_len=256, seed=42, label="HOLDOUT", device="auto"):
    device = resolve_device(device)
    resolved_file = resolve_path(file_path)
    if not os.path.exists(resolved_file):
        raise FileNotFoundError(f"CRITICAL ERROR: {label} file '{file_path}' does not exist! "
                                f"Silently falling back to training data is strictly prohibited.")

    with open(resolved_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = torch.tensor(enc.encode_ordinary(text), dtype=torch.long, device=device)
    total_tokens_in_file = len(tokens)

    required_tokens = seq_len + 1
    if total_tokens_in_file < required_tokens:
        raise ValueError(f"Corpus '{file_path}' has only {total_tokens_in_file} tokens, less than required {required_tokens}.")

    max_start = total_tokens_in_file - seq_len - 1
    if total_tokens_in_file >= num_windows * (seq_len + 1):
        window_starts = [i * seq_len for i in range(num_windows)]
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        window_starts = torch.randint(0, max_start, (num_windows,), generator=g).tolist()

    losses = []
    has_nan_or_inf = False

    for start_idx in window_starts:
        x = tokens[start_idx : start_idx + seq_len].unsqueeze(0)
        y = tokens[start_idx + 1 : start_idx + seq_len + 1].unsqueeze(0)

        with get_autocast(device):
            logits, loss = model(x, targets=y, persist_state=False)

        val = loss.item()
        if math.isnan(val) or math.isinf(val):
            has_nan_or_inf = True
        losses.append(val)

    mean_loss = statistics.mean(losses)
    std_loss = statistics.stdev(losses) if len(losses) > 1 else 0.0
    ppl = math.exp(min(mean_loss, 100.0))
    total_eval_tokens = num_windows * seq_len

    return {
        "label": label,
        "file": resolved_file,
        "corpus_total_tokens": total_tokens_in_file,
        "windows_evaluated": len(losses),
        "tokens_evaluated": total_eval_tokens,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "perplexity": ppl,
        "has_nan_inf": has_nan_or_inf,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    enc = ENCODING
    device = resolve_device(args.device)

    # 1. Load production model
    model, ckpt_path, trained_step = load_jarvis_model(args.ckpt, device=device)

    # 2. Interactive Terminal Mode
    if args.interactive:
        interactive_terminal(model, enc, args, ckpt_path, device=device)
        return

    # 3. Single Prompt Evaluation Mode
    if args.prompt:
        prompt_text = args.prompt
        if not getattr(args, "raw", False) and not (prompt_text.strip().startswith("User:") or prompt_text.strip().startswith("System:")):
            prompt_text = f"User: {prompt_text.strip()}\nAssistant: "
        print(f"Prompt: {prompt_text}\n")
        print("Response:")
        generate_streaming(
            model=model,
            enc=enc,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            ckpt_path=ckpt_path
        )
        return

    eval_candidates = [
        args.eval_file,
        "fresh_holdout.txt",
        "assistant_sft_tokens_holdout.txt",
        "code_instruction_sft_tokens_holdout.txt",
    ]
    eval_file_resolved = None
    for cand in eval_candidates:
        if cand:
            r = resolve_path(cand)
            if r and os.path.exists(r):
                eval_file_resolved = r
                break
    if not eval_file_resolved:
        eval_file_resolved = resolve_path("fresh_holdout.txt")
    train_file_resolved = resolve_path(args.train_file, "data.txt")

    print("=" * 90)
    print(f"  JARVIS 606M POST-TRAINING CORPUS EVALUATION")
    print(f"  Target Checkpoint: {ckpt_path} (Step {trained_step})")
    print(f"  Holdout Corpus:    {eval_file_resolved}")
    print(f"  Training Corpus:   {train_file_resolved}")
    print(f"  Eval Windows:      {args.num_windows} windows x {args.seq_len} tokens")
    print("=" * 90)

    train_res = None
    if not args.skip_train_eval and os.path.exists(train_file_resolved):
        print(f"\nEvaluating TRAINING CORPUS: {train_file_resolved}...", flush=True)
        train_res = evaluate_corpus(model, train_file_resolved, enc,
                                    num_windows=args.num_windows, seq_len=args.seq_len,
                                    seed=args.seed, label="TRAINING LOSS", device=device)

    print(f"\nEvaluating HOLDOUT CORPUS: {eval_file_resolved}...", flush=True)
    holdout_res = evaluate_corpus(model, eval_file_resolved, enc,
                                  num_windows=args.num_windows, seq_len=args.seq_len,
                                  seed=args.seed, label="HOLDOUT LOSS", device=device)

    print("\n" + "=" * 90)
    print("  QUANTITATIVE EVALUATION SUMMARY")
    print("=" * 90)
    header = f"{'Metric':<32} | {'TRAINING LOSS':<28} | {'HOLDOUT LOSS':<26}"
    print(header)
    print("-" * 90)
    t_mean = f"{train_res['mean_loss']:.4f}" if train_res else "N/A"
    h_mean = f"{holdout_res['mean_loss']:.4f}"
    print(f"{'Mean Cross-Entropy Loss':<32} | {t_mean:<28} | {h_mean:<26}")
    t_ppl = f"{train_res['perplexity']:.2f}" if train_res else "N/A"
    h_ppl = f"{holdout_res['perplexity']:.2f}"
    print(f"{'Perplexity (exp(loss))':<32} | {t_ppl:<28} | {h_ppl:<26}")
    print("=" * 90)


if __name__ == "__main__":
    main()
