import os
import sys
import time
import statistics
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

def percentile(data, p):
    sorted_d = sorted(data)
    idx = int(round(p * (len(sorted_d) - 1)))
    return sorted_d[idx]

def run_clean_baseline():
    print("=" * 100, flush=True)
    print("JARVIS COMPONENT 6: CLEAN AUTHORITATIVE PERFORMANCE BASELINE", flush=True)
    print("=" * 100, flush=True)
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # 1. System environment verification
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM, SM {props.major}.{props.minor}, {props.multi_processor_count} SMs)", flush=True)
    print(f"PyTorch: {torch.__version__} | CUDA: 13.3 | Architecture compute_120/sm_120", flush=True)
    
    B, T, C, H, E, top_k, hidden_dim, chunk_size, num_layers, vocab_size, vocab_pad, accum_steps = (
        4, 512, 1024, 16, 4, 2, 2048, 64, 24, 50257, 50304, 2
    )
    tokens_per_update = B * T * accum_steps # 4096 real tokens
    
    print(f"Workload: B={B}, T={T}, accum={accum_steps} -> {tokens_per_update} real tokens/update", flush=True)
    print(f"Model: {num_layers} layers, C={C}, H={H}, E={E}, top_k={top_k}, hidden={hidden_dim}, vocab={vocab_size}", flush=True)
    
    # 2. Allocate and bind persistent model parameters
    ref_model = Jarvis(
        vocab_size=vocab_size, d_model=C, n_layers=num_layers, n_heads=H,
        num_experts=E, top_k=top_k, max_seq_len=T, use_cuda_attn=True, use_cuda_moe=True
    ).to(device=device, dtype=dtype)
    
    # The native CUDA engine holds raw pointers to these buffers throughout training.
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
    
    # 3. Clean up reference PyTorch model container, keeping persistent parameter buffers intact
    del ref_model
    torch.cuda.empty_cache()
    
    lr = 1e-4
    
    # -------------------------------------------------------------
    # 4. Eager Mode Baseline Measurement (20 Warmup, 100 Measured)
    # -------------------------------------------------------------
    print("\n--- [1/2] EAGER MODE BENCHMARK (100 Updates) ---", flush=True)
    print("Warmup 10 eager steps...", flush=True)
    for _ in range(10):
        nce.train_step(lr)
    torch.cuda.synchronize()
    
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    
    eager_times_ms = []
    num_measured = 100
    print(f"Benchmarking {num_measured} eager updates with CUDA Events...", flush=True)
    for i in range(num_measured):
        start_ev.record()
        nce.train_step(lr)
        end_ev.record()
        end_ev.synchronize()
        eager_times_ms.append(start_ev.elapsed_time(end_ev))
        if (i + 1) % 20 == 0:
            print(f"  Eager Step {i+1:3d}/{num_measured}: current = {eager_times_ms[-1]:.2f}ms, running mean = {statistics.mean(eager_times_ms):.2f}ms", flush=True)
            
    eager_mean = statistics.mean(eager_times_ms)
    eager_median = statistics.median(eager_times_ms)
    eager_p90 = percentile(eager_times_ms, 0.90)
    eager_p99 = percentile(eager_times_ms, 0.99)
    eager_tok_s = tokens_per_update / (eager_mean / 1000.0)
    
    # -------------------------------------------------------------
    # 5. CUDA Graph Baseline Measurement (20 Warmup, 100 Measured)
    # -------------------------------------------------------------
    print("\n--- [2/2] CUDA GRAPH BASELINE BENCHMARK (100 Updates) ---", flush=True)
    print("Capturing full 24-layer DAG into CUDA Graph...", flush=True)
    t0 = time.time()
    nce.capture_full_graph(lr)
    torch.cuda.synchronize()
    print(f"Graph captured in {time.time() - t0:.2f}s.", flush=True)
    
    print("Warmup 10 graph replay steps...", flush=True)
    for _ in range(10):
        nce.replay_graph()
    torch.cuda.synchronize()
    
    graph_times_ms = []
    print(f"Benchmarking {num_measured} CUDA Graph replay updates with CUDA Events...", flush=True)
    for i in range(num_measured):
        start_ev.record()
        nce.replay_graph()
        end_ev.record()
        end_ev.synchronize()
        graph_times_ms.append(start_ev.elapsed_time(end_ev))
        if (i + 1) % 20 == 0:
            print(f"  Graph Step {i+1:3d}/{num_measured}: current = {graph_times_ms[-1]:.2f}ms, running mean = {statistics.mean(graph_times_ms):.2f}ms", flush=True)
            
    graph_mean = statistics.mean(graph_times_ms)
    graph_median = statistics.median(graph_times_ms)
    graph_p90 = percentile(graph_times_ms, 0.90)
    graph_p99 = percentile(graph_times_ms, 0.99)
    graph_tok_s = tokens_per_update / (graph_mean / 1000.0)
    
    ws_bytes = nce.get_workspace_memory_bytes()
    
    # -------------------------------------------------------------
    # 6. Final Baseline Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 100, flush=True)
    print("OFFICIAL COMPONENT 6 AUTHORITATIVE PERFORMANCE BASELINE RESULTS", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Metric':<30} | {'Eager Mode':<20} | {'CUDA Graph Mode':<20}", flush=True)
    print("-" * 100, flush=True)
    print(f"{'Mean Step Time (ms)':<30} | {eager_mean:<20.2f} | {graph_mean:<20.2f}", flush=True)
    print(f"{'Median Step Time (ms)':<30} | {eager_median:<20.2f} | {graph_median:<20.2f}", flush=True)
    print(f"{'p90 Step Time (ms)':<30} | {eager_p90:<20.2f} | {graph_p90:<20.2f}", flush=True)
    print(f"{'p99 Step Time (ms)':<30} | {eager_p99:<20.2f} | {graph_p99:<20.2f}", flush=True)
    print(f"{'Sustained Throughput (tok/s)':<30} | {eager_tok_s:<20.1f} | {graph_tok_s:<20.1f}", flush=True)
    print(f"{'Tokens per Update':<30} | {tokens_per_update:<20} | {tokens_per_update:<20}", flush=True)
    print(f"{'Static Workspace VRAM':<30} | {ws_bytes / (1024**3):<16.3f} GiB | {ws_bytes / (1024**3):<16.3f} GiB", flush=True)
    print("=" * 100, flush=True)
    
    nce.cleanup_full_engine()

if __name__ == "__main__":
    run_clean_baseline()
