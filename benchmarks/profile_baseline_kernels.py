import os
import sys
import torch
import numpy as np

WORKSPACE_ROOT = os.path.abspath(".")
JARVIS_ENGINE = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
for p in [WORKSPACE_ROOT, JARVIS_ENGINE, os.path.join(WORKSPACE_ROOT, "associative_attention_cuda"), os.path.join(WORKSPACE_ROOT, "sparse_model_cuda")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

for p in [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"]:
    if os.path.exists(p) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(p)

from jarvis_model import Jarvis
import native_cuda_engine as nce

def profile_locked_baseline():
    print("=" * 115, flush=True)
    print("JARVIS COMPONENT 6: PROFILING LOCKED BASELINE KERNEL EXECUTION", flush=True)
    print("Device: NVIDIA GeForce RTX 5070 12GB (Blackwell SM120, 48 SMs)")
    print("Workload: B=4, T=512, accum=2 (4096 real tokens/update), 24 layers, 4 experts, top_k=2")
    print("=" * 115, flush=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, T, C, H, E, top_k, hidden_dim, chunk_size, num_layers, vocab_size, vocab_pad, accum_steps = (
        4, 512, 1024, 16, 4, 2, 2048, 64, 24, 50257, 50304, 2
    )

    torch.manual_seed(42)
    ref_model = Jarvis(
        vocab_size=vocab_size, d_model=C, n_layers=num_layers, n_heads=H,
        num_experts=E, top_k=top_k, max_seq_len=T, use_cuda_attn=True, use_cuda_moe=True
    ).to(device=device, dtype=dtype)

    tok_emb_w = ref_model.tok_emb.weight.data.clone()
    final_norm_w = ref_model.final_norm.weight.data.clone()
    lm_head_pad = torch.zeros(vocab_pad, C, device=device, dtype=dtype)
    lm_head_pad[:vocab_size, :] = ref_model.lm_head.weight.data.clone()

    nce.init_full_engine(B, T, C, H, E, top_k, hidden_dim, chunk_size, num_layers, vocab_size, vocab_pad, accum_steps)
    nce.bind_global_parameters(tok_emb_w, final_norm_w, lm_head_pad)

    layer_params = []
    for l, block in enumerate(ref_model.blocks):
        norm1_w = block.norm1.weight.data.clone()
        qkv_w = torch.cat([block.attn.q_proj.weight.data, block.attn.k_proj.weight.data, block.attn.v_proj.weight.data], dim=0).clone()
        out_proj_w = block.attn.out_proj.weight.data.clone()
        norm2_w = block.norm2.weight.data.clone()
        router_w = block.moe.router.weight.data.clone()
        w1_list = [block.moe.w1[e].weight.data.clone() for e in range(E)]
        w2_list = [block.moe.w2[e].weight.data.clone() for e in range(E)]
        gamma_raw = block.attn.gamma_raw.data.clone().float()
        var_scale = block.liquid.var_scale.data.clone().float()

        nce.bind_layer_parameters(
            l, norm1_w, qkv_w, out_proj_w, norm2_w,
            router_w,
            w1_list[0], w1_list[1], w1_list[2], w1_list[3],
            w2_list[0], w2_list[1], w2_list[2], w2_list[3],
            gamma_raw, var_scale
        )
        layer_params.append({
            "norm1_w": norm1_w, "qkv_w": qkv_w, "out_proj_w": out_proj_w, "norm2_w": norm2_w,
            "router_w": router_w, "w1": w1_list, "w2": w2_list, "gamma_raw": gamma_raw, "var_scale": var_scale
        })

    inp = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.int32)
    tgt = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.int32)
    nce.set_inputs(inp, tgt)

    del ref_model
    torch.cuda.empty_cache()

    lr = 1e-4

    print("Warmup 10 steps...", flush=True)
    for _ in range(10):
        nce.train_step(lr)
    torch.cuda.synchronize()

    trials = 30
    print(f"Collecting PyTorch CUDA Activity Trace across {trials} updates...", flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof:
        for _ in range(trials):
            nce.train_step(lr)
        torch.cuda.synchronize()

    key_averages = prof.key_averages()
    print("\n" + "=" * 115, flush=True)
    print("TOP 30 KERNELS RANKED BY TOTAL CUDA TIME (LOCKED BASELINE):", flush=True)
    print("=" * 115, flush=True)
    print(key_averages.table(sort_by="cuda_time_total", row_limit=30), flush=True)

    nce.cleanup_full_engine()

if __name__ == "__main__":
    profile_locked_baseline()
