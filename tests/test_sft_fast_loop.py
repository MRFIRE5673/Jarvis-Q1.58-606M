"""
test_sft_fast_loop.py — Research & Experimental Test Suite for SFT & 4-Layer Stacking
=====================================================================================
1. TEST 1: SFT Masked Training Convergence on 4-Layer Miniature Model
   - Verifies train_step_masked runs correctly without graph capture.
   - Compares convergence rate of LR=2e-5 vs LR=1.5e-4 on Assistant SFT tokens.
2. TEST 2: Progressive Layer Stacking / Expansion (4 -> 24 Layers)
   - Tests weight transfer/stacking from 4 layers to 24 layers (Solar-style depth expansion).
   - Verifies 24-layer engine initializes and executes forward/backward steps cleanly.
"""

import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from jarvis_engine.cuda_engine import eng, _ENGINE_AVAILABLE


def test_sft_masked_convergence_4layer():
    """Verify SFT training dynamics and learning rate speedup on 4-layer engine."""
    assert _ENGINE_AVAILABLE and eng is not None, "Native CUDA engine required"

    sft_data_path = os.path.join(ROOT, "data", "assistant_sft_tokens.npy")
    sft_mask_path = os.path.join(ROOT, "data", "assistant_sft_tokens_mask.npy")
    if not (os.path.exists(sft_data_path) and os.path.exists(sft_mask_path)):
        print("SFT dataset files not found, skipping.")
        return

    tok = np.load(sft_data_path, mmap_mode="r")
    mask = np.load(sft_mask_path, mmap_mode="r")

    B, T, ACC = 8, 512, 2
    NT = B * T * ACC  # 8192 tokens/step
    L_mini = 4

    # Run comparison: 15 steps with LR=2e-5 vs 15 steps with LR=1.5e-4
    losses = {}
    for lr, label in [(2e-5, "slow_lr"), (1.5e-4, "fast_lr")]:
        eng.destroy()
        eng.init(
            B, T, 1024, 16, L_mini, 4, 2, 2048, 50257, 50304, 64, ACC,
            lr, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0,
            10, 100, 0.1, 0
        )

        # Initialize pseudo-random weights
        rng = np.random.default_rng(42)
        n = lambda *s: (rng.standard_normal(s, dtype=np.float32) * 0.02)
        eng.load_weight("emb", n(50304, 1024))
        eng.load_weight("final_norm", np.ones(1024, np.float32))
        for l in range(L_mini):
            eng.load_weight(f"qkv.{l}", n(3072, 1024))
            eng.load_weight(f"out.{l}", n(1024, 1024))
            eng.load_weight(f"w1.{l}", n(4 * 2048, 1024))
            eng.load_weight(f"w2.{l}", n(4 * 1024, 2048))
            eng.load_weight(f"norm1.{l}", np.ones(1024, np.float32))
            eng.load_weight(f"norm2.{l}", np.ones(1024, np.float32))
            eng.load_weight(f"gamma.{l}", np.full(16, np.log(0.95), np.float32))
            eng.load_weight(f"router.{l}", n(4, 1024))

        step_losses = []
        for step in range(15):
            start = step * NT
            batch_tok = np.array(tok[start:start + NT], dtype=np.int32)
            batch_mask = np.array(mask[start:start + NT], dtype=np.uint8)
            active_targets = int(np.count_nonzero(batch_mask.reshape(B * ACC, T)[:, 1:]))
            if active_targets == 0:
                continue

            loss = eng.train_step_masked(batch_tok, batch_mask, active_targets)
            assert np.isfinite(loss), f"Non-finite loss at step {step}"
            step_losses.append(loss)

        losses[label] = step_losses
        print(f"[{label} (LR={lr})] Initial loss: {step_losses[0]:.4f} -> Step 15 loss: {step_losses[-1]:.4f} (Delta: {step_losses[0] - step_losses[-1]:.4f})")

    # Fast LR should achieve faster descent on SFT masked tokens
    delta_slow = losses["slow_lr"][0] - losses["slow_lr"][-1]
    delta_fast = losses["fast_lr"][0] - losses["fast_lr"][-1]
    print(f"Loss drop: Fast LR={delta_fast:.4f} vs Slow LR={delta_slow:.4f}")
    assert delta_fast > delta_slow, "Higher LR (1.5e-4) should converge faster than 2e-5"
    eng.destroy()


def test_progressive_layer_stacking():
    """Verify progressive layer expansion: replicate/stack 4-layer weights into 24-layer engine."""
    assert _ENGINE_AVAILABLE and eng is not None, "Native CUDA engine required"

    # 1. Simulate 4 trained layers
    L_mini = 4
    rng = np.random.default_rng(123)
    n = lambda *s: (rng.standard_normal(s, dtype=np.float32) * 0.02)
    mini_weights = {}
    mini_weights["emb"] = n(50304, 1024)
    mini_weights["final_norm"] = np.ones(1024, np.float32)
    for l in range(L_mini):
        mini_weights[f"qkv.{l}"] = n(3072, 1024)
        mini_weights[f"out.{l}"] = n(1024, 1024)
        mini_weights[f"w1.{l}"] = n(4 * 2048, 1024)
        mini_weights[f"w2.{l}"] = n(4 * 1024, 2048)
        mini_weights[f"norm1.{l}"] = np.ones(1024, np.float32)
        mini_weights[f"norm2.{l}"] = np.ones(1024, np.float32)
        mini_weights[f"gamma.{l}"] = np.full(16, np.log(0.95), np.float32)
        mini_weights[f"router.{l}"] = n(4, 1024)

    # 2. Stack/expand 4 layers to 24 layers (6x repeat or progressive graft)
    L_full = 24
    B, T, ACC = 8, 512, 2
    eng.destroy()
    eng.init(
        B, T, 1024, 16, L_full, 4, 2, 2048, 50257, 50304, 64, ACC,
        1e-4, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0,
        10, 100, 0.1, 0
    )

    eng.load_weight("emb", mini_weights["emb"])
    eng.load_weight("final_norm", mini_weights["final_norm"])
    for l_target in range(L_full):
        l_source = l_target % L_mini  # Progressive depth repeat
        eng.load_weight(f"qkv.{l_target}", mini_weights[f"qkv.{l_source}"])
        eng.load_weight(f"out.{l_target}", mini_weights[f"out.{l_source}"])
        eng.load_weight(f"w1.{l_target}", mini_weights[f"w1.{l_source}"])
        eng.load_weight(f"w2.{l_target}", mini_weights[f"w2.{l_source}"])
        eng.load_weight(f"norm1.{l_target}", mini_weights[f"norm1.{l_source}"])
        eng.load_weight(f"norm2.{l_target}", mini_weights[f"norm2.{l_source}"])
        eng.load_weight(f"gamma.{l_target}", mini_weights[f"gamma.{l_source}"])
        eng.load_weight(f"router.{l_target}", mini_weights[f"router.{l_source}"])

    # 3. Verify forward/backward step on the 24-layer stacked model
    sft_data_path = os.path.join(ROOT, "data", "assistant_sft_tokens.npy")
    sft_mask_path = os.path.join(ROOT, "data", "assistant_sft_tokens_mask.npy")
    tok = np.load(sft_data_path, mmap_mode="r")
    mask = np.load(sft_mask_path, mmap_mode="r")
    NT = B * T * ACC
    batch_tok = np.array(tok[:NT], dtype=np.int32)
    batch_mask = np.array(mask[:NT], dtype=np.uint8)
    active_targets = int(np.count_nonzero(batch_mask.reshape(B * ACC, T)[:, 1:]))

    loss = eng.train_step_masked(batch_tok, batch_mask, active_targets)
    assert np.isfinite(loss), f"Stacked 24-layer model produced non-finite loss: {loss}"
    print(f"[Stacked 24-layer Model] Successfully executed step! Masked loss: {loss:.4f}")
    eng.destroy()


if __name__ == "__main__":
    test_sft_masked_convergence_4layer()
    test_progressive_layer_stacking()
    print("ALL SFT & LAYER STACKING TESTS PASSED!")
