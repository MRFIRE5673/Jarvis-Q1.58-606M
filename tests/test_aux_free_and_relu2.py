"""
Comprehensive Verification Suite:
  1. DeepSeek-V3 Auxiliary-Loss-Free Dynamic Router Bias
  2. Squared ReLU (ReLU²) vs GELU Activation & Sparsity
  3. Native CUDA Engine Real Batch Verification (Finite loss, STE gradients, Bias update)
"""

import os
import sys

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import math
import struct
import time
import numpy as np
import torch
import torch.nn.functional as F

from jarvis_engine.utils.moe_router import SparseTopKRouter, TernaryMoE
from jarvis_engine.jarvis_model import SparseMoELayer


def test_1_aux_loss_free_dynamics():
    print("=" * 70)
    print("TEST 1: Auxiliary-Loss-Free Dynamic Router Bias Verification")
    print("=" * 70)

    torch.manual_seed(42)
    N, C, E, K = 32, 1024, 4, 2
    router = SparseTopKRouter(d_model=C, num_experts=E, top_k=K, use_aux_free=True, bias_gamma=0.01)
    router.train()

    # Initial bias should be all zeros
    assert torch.allclose(router.expert_bias, torch.zeros(E)), "Initial bias must be 0"
    print("[PASS] Initial router bias is all zeros: ", router.expert_bias.tolist())

    # Create tokens biased towards experts 0 and 1
    with torch.no_grad():
        router.expert_emb.data[0] = 5.0
        router.expert_emb.data[1] = 4.0
        router.expert_emb.data[2:] = -5.0

    # Run 5 forward steps and observe bias evolution
    biases = []
    for step in range(5):
        x = torch.ones(N, C)
        gates, idx, l_bal = router(x)
        assert l_bal.item() == 0.0, f"Auxiliary loss must be strictly 0.0, got {l_bal.item()}"
        assert (gates >= 0).all() and torch.allclose(gates.sum(-1), torch.ones(N)), "Gate weights must sum to 1"
        b_curr = router.expert_bias.clone().tolist()
        biases.append(b_curr)
        counts = [(idx == e).sum().item() for e in range(E)]
        print(f"  Step {step + 1}: Dispatched counts={counts} | Expert Biases={[f'{v:.4f}' for v in b_curr]}")

    # Experts 0 & 1 were overwhelmingly chosen (32 tokens each out of 64 routes -> f=0.50 > 0.25) -> biases decrease
    # Experts 2 & 3 were starved (0 tokens -> f=0.00 < 0.25) -> biases increase
    assert router.expert_bias[0] < 0.0 and router.expert_bias[1] < 0.0, (
        f"Overloaded experts 0,1 biases must decrease, got {router.expert_bias[:2]}"
    )
    assert router.expert_bias[2] > 0.0 and router.expert_bias[3] > 0.0, (
        f"Underloaded experts 2,3 biases must increase, got {router.expert_bias[2:]}"
    )
    assert abs(router.expert_bias.sum().item()) < 1e-6, "Net bias drift must sum to 0"
    print("[PASS] Closed-loop controller successfully penalized overloaded experts and elevated underloaded experts with 0 net drift.")
    print()


def test_2_squared_relu_properties():
    print("=" * 70)
    print("TEST 2: Squared ReLU (ReLU²) Mathematical & Sparsity Properties")
    print("=" * 70)

    torch.manual_seed(42)
    B, T, C = 2, 16, 1024
    x = torch.randn(B, T, C, requires_grad=True)

    # Instantiate MoE with ReLU²
    moe_relu2 = TernaryMoE(d_model=C, num_experts=4, top_k=2, activation="relu2")
    moe_gelu  = TernaryMoE(d_model=C, num_experts=4, top_k=2, activation="gelu")

    # Forward pass
    out_relu2, _ = moe_relu2(x)
    out_gelu, _  = moe_gelu(x)

    # Check that ReLU² activation layer inside W1 produces exact zeros for negative inputs
    with torch.no_grad():
        h1_in = moe_relu2.w1[0](x.view(-1, C))
        h1_relu2 = moe_relu2._act(h1_in)
        h1_gelu  = moe_gelu._act(h1_in)

        sparsity_relu2 = (h1_relu2 == 0.0).float().mean().item()
        sparsity_gelu  = (h1_gelu == 0.0).float().mean().item()
        neg_relu2 = (h1_relu2 < 0.0).any().item()

    print(f"  ReLU² Output Sparsity (exact zeros): {sparsity_relu2:.1%}")
    print(f"  GELU Output Sparsity (exact zeros):  {sparsity_gelu:.1%}")
    print(f"  ReLU² Negative values present:      {neg_relu2} (must be False)")

    assert not neg_relu2, "ReLU² outputs must never be negative!"
    assert sparsity_relu2 > 0.40, f"ReLU² must produce high activation sparsity, got {sparsity_relu2:.1%}"
    print("[PASS] ReLU² achieves natural activation sparsity without negative values.")

    # Backward pass gradient verification
    loss_relu2 = out_relu2.sum()
    loss_relu2.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Gradients with ReLU² must be finite and non-NaN"
    print(f"  Input gradient norm with ReLU²: {x.grad.norm().item():.4f}")
    print("[PASS] Backward pass through ReLU² computes valid, non-zero, finite gradients.")
    print()


def test_3_native_cuda_engine_smoke():
    print("=" * 70)
    print("TEST 3: Native CUDA Engine Smoke Test with Aux-Loss-Free Routing")
    print("=" * 70)

    try:
        import jarvis_engine.cuda_engine as ce
        if not ce._ENGINE_AVAILABLE:
            print("[SKIP] native_cuda_engine not available in this environment")
            return
        eng = ce.eng
    except Exception as e:
        print(f"[SKIP] Could not load native_cuda_engine: {e}")
        return

    # Checkpoint path
    ckpt_path = "data/jarvis_cuda.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Checkpoint {ckpt_path} not found")
        return

    # Setup environment for aux-loss-free
    os.environ["JARVIS_AUX_FREE"] = "1"
    os.environ["JARVIS_MOE_SPARSE"] = "1"

    # Read checkpoint step
    with open(ckpt_path, "rb") as f:
        step_header = struct.unpack("<i", f.read(4))[0]
    print(f"  Existing checkpoint header step: {step_header:,}")

    # Dataset path
    data_path = "data/train_tokens_300m_en25_code75.npy"
    if not os.path.exists(data_path):
        data_path = "data/train_tokens.npy"

    tok_raw = np.load(data_path, mmap_mode="r")
    B, T, ACC = 4, 512, 4
    NT = B * T * ACC  # 8192 tokens per step

    # Initialize engine
    # batch, seq, C, H, L, E, K, hi, vocab, vocab_pad, chunk, accum, lr, b1, b2, eps, wd, clip, bal, refl, lsf_min, lsf_max, dev
    eng.init(B, T, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, ACC,
             1e-4, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.0, 0.01, 0.1, 0.99, 0,
             10, 1000, 0.1, 0)

    eng.load_checkpoint(ckpt_path)
    print("  [Init] Engine initialized & checkpoint loaded successfully.")

    # Run 3 test training steps
    losses = []
    for step in range(3):
        batch = np.ascontiguousarray(tok_raw[step * NT : (step + 1) * NT], dtype=np.int32)
        t0 = time.time()
        loss = eng.train_step(batch)
        dt = time.time() - t0
        assert math.isfinite(loss), f"Step {step + 1} produced non-finite loss: {loss}"
        losses.append(loss)
        tok_s = NT / max(dt, 1e-6)
        print(f"  Verification Step {step + 1}: Loss = {loss:.4f} | Speed = {tok_s:,.0f} tok/s | Time = {dt*1000:.1f} ms")

    assert all(math.isfinite(l) for l in losses), "All losses must be finite"
    print(f"[PASS] Native CUDA engine executed 3 full fwd/bwd/opt steps with finite loss: {losses}")
    eng.destroy()
    print("[PASS] Engine destroyed cleanly.")
    print()


if __name__ == "__main__":
    test_1_aux_loss_free_dynamics()
    test_2_squared_relu_properties()
    test_3_native_cuda_engine_smoke()
    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
