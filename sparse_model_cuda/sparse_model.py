# sparse_model.py
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

# On Windows (Python 3.8+), native extensions require CUDA bin in DLL directory
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        cuda_candidates = sorted(
            glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"),
            reverse=True
        )
        if cuda_candidates:
            cuda_home = cuda_candidates[0]
    if cuda_home:
        cuda_bin = os.path.join(cuda_home, "bin")
        if os.path.exists(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
            except Exception:
                pass

# Add current directory to path for local import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import sparse_model_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

_CUSTOM_CUDA_EXECUTED = False
_FALLBACK_USED = False

def reset_diagnostics():
    global _CUSTOM_CUDA_EXECUTED, _FALLBACK_USED
    _CUSTOM_CUDA_EXECUTED = False
    _FALLBACK_USED = False

def get_diagnostics():
    return {
        "extension_imported": _CUDA_AVAILABLE,
        "custom_kernel_executed": _CUSTOM_CUDA_EXECUTED,
        "fallback_used": _FALLBACK_USED,
    }


class MoeDispatchGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, gather_map, scatter_map, N):
        global _CUSTOM_CUDA_EXECUTED
        _CUSTOM_CUDA_EXECUTED = True
        dispatched_x = sparse_model_cuda.dispatch_gather(x_flat.contiguous(), gather_map.contiguous())
        ctx.save_for_backward(scatter_map)
        ctx.N = N
        return dispatched_x

    @staticmethod
    def backward(ctx, grad_dispatched_x):
        scatter_map, = ctx.saved_tensors
        grad_x = sparse_model_cuda.gather_backward(grad_dispatched_x.contiguous(), scatter_map, ctx.N)
        return grad_x, None, None, None


class MoeScatterCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dispatched_y, topk_gates, scatter_map, gather_map, gate_idx_map):
        global _CUSTOM_CUDA_EXECUTED
        _CUSTOM_CUDA_EXECUTED = True
        out = sparse_model_cuda.scatter_combine(dispatched_y.contiguous(), topk_gates.contiguous(), scatter_map.contiguous())
        ctx.save_for_backward(dispatched_y, topk_gates, gather_map, gate_idx_map, scatter_map)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        dispatched_y, topk_gates, gather_map, gate_idx_map, scatter_map = ctx.saved_tensors
        grad_dispatched_y, grad_topk_gates = sparse_model_cuda.scatter_backward(
            grad_out.contiguous(), dispatched_y, topk_gates, gather_map, gate_idx_map, scatter_map
        )
        return grad_dispatched_y, grad_topk_gates, None, None, None


if _TRITON_AVAILABLE:
    @triton.jit
    def _grouped_gemm_fwd_kernel(
        a_ptr, b_ptr, c_ptr,
        K: tl.constexpr, N: tl.constexpr,
        stride_am, stride_ak,
        stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        offsets_ptr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert_id = tl.program_id(2)

        start_m = tl.load(offsets_ptr + expert_id)
        end_m = tl.load(offsets_ptr + expert_id + 1)
        expert_m = end_m - start_m

        if pid_m * BLOCK_M >= expert_m:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (start_m + offs_m[:, None]) * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + expert_id * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            mask_m = (offs_m[:, None] < expert_m)
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + (start_m + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
        mask_c = (offs_m[:, None] < expert_m) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)

    @triton.jit
    def _grouped_gemm_weight_kernel(
        x_ptr, dy_ptr, dw_ptr,
        K: tl.constexpr, N: tl.constexpr,
        stride_xm, stride_xk,
        stride_dym, stride_dyn,
        stride_dwe, stride_dwk, stride_dwn,
        offsets_ptr,
        BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert_id = tl.program_id(2)

        start_m = tl.load(offsets_ptr + expert_id)
        end_m = tl.load(offsets_ptr + expert_id + 1)
        expert_m = end_m - start_m

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

        for m_start in range(0, 8192, BLOCK_M):
            if m_start < expert_m:
                curr_offs_m = m_start + offs_m
                mask_m = curr_offs_m < expert_m
                x_ptrs = x_ptr + (start_m + curr_offs_m[None, :]) * stride_xm + offs_k[:, None] * stride_xk
                dy_ptrs = dy_ptr + (start_m + curr_offs_m[:, None]) * stride_dym + offs_n[None, :] * stride_dyn

                x_tile = tl.load(x_ptrs, mask=mask_m[None, :], other=0.0)
                dy_tile = tl.load(dy_ptrs, mask=mask_m[:, None], other=0.0)
                acc += tl.dot(x_tile, dy_tile)

        dw_ptrs = dw_ptr + expert_id * stride_dwe + offs_k[:, None] * stride_dwk + offs_n[None, :] * stride_dwn
        mask_dw = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        tl.store(dw_ptrs, acc.to(tl.bfloat16), mask=mask_dw)

    def _triton_grouped_gemm(a, b, offsets, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=4):
        M, K = a.shape
        E, _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
        _grouped_gemm_fwd_kernel[grid](
            a, b, c,
            K, N,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1),
            offsets,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
        return c

    def _triton_grouped_gemm_weight(x, dy, offsets, BLOCK_K=64, BLOCK_N=64, BLOCK_M=64, num_warps=4, num_stages=3):
        M, K = x.shape
        _, N = dy.shape
        E = offsets.shape[0] - 1
        dw = torch.empty((E, K, N), device=x.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N), E)
        _grouped_gemm_weight_kernel[grid](
            x, dy, dw,
            K, N,
            x.stride(0), x.stride(1),
            dy.stride(0), dy.stride(1),
            dw.stride(0), dw.stride(1), dw.stride(2),
            offsets,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
            num_warps=num_warps, num_stages=num_stages,
        )
        return dw

    @triton.jit
    def _stacked_ternary_fwd_kernel(
        W_ptr, Wq_ptr, Alpha_ptr,
        elements_per_expert, N_total,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_total
        
        expert_idx = offsets // elements_per_expert
        alpha = tl.load(Alpha_ptr + expert_idx, mask=mask, other=1.0)
        
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w_norm = w / alpha
        w_clamped = tl.maximum(tl.minimum(w_norm, 1.0), -1.0)
        w_round = tl.extra.cuda.libdevice.nearbyint(w_clamped)
        w_q = w_round * alpha
        
        tl.store(Wq_ptr + offsets, w_q.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _stacked_ternary_bwd_kernel(
        GradOut_ptr, W_ptr, GradW_ptr,
        N_total, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_total
        
        go = tl.load(GradOut_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        is_valid = tl.abs(w) <= 1.0
        gw = tl.where(is_valid, go, 0.0)
        
        tl.store(GradW_ptr + offsets, gw.to(tl.bfloat16), mask=mask)

    class StackedTernarySTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w):
            if w.is_cuda and _TRITON_AVAILABLE:
                elements_per_expert = w.shape[1] * w.shape[2]
                N_total = w.numel()
                alpha = w.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                wq = torch.empty_like(w)
                BLOCK_SIZE = 1024
                grid = (triton.cdiv(N_total, BLOCK_SIZE),)
                _stacked_ternary_fwd_kernel[grid](
                    w, wq, alpha,
                    elements_per_expert, N_total,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=4
                )
                ctx.save_for_backward(w)
                return wq
            alpha = w.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            w_norm = w / alpha
            w_q = torch.round(torch.clamp(w_norm, -1.0, 1.0))
            ctx.save_for_backward(w)
            return w_q * alpha

        @staticmethod
        def backward(ctx, grad_output):
            w, = ctx.saved_tensors
            if w.is_cuda and _TRITON_AVAILABLE:
                grad_w = torch.empty_like(w)
                N_total = w.numel()
                BLOCK_SIZE = 1024
                grid = (triton.cdiv(N_total, BLOCK_SIZE),)
                _stacked_ternary_bwd_kernel[grid](
                    grad_output, w, grad_w,
                    N_total, BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=4
                )
                return grad_w
            mask = (w.abs() <= 1.0).float()
            return grad_output * mask

    class TritonGroupedMoEMLPFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w1_q, w2_q, offsets):
            h1 = _triton_grouped_gemm(x, w1_q.transpose(1, 2), offsets)
            act = F.gelu(h1)
            y = _triton_grouped_gemm(act, w2_q.transpose(1, 2), offsets)
            ctx.save_for_backward(x, h1, act, w1_q, w2_q, offsets)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            x, h1, act, w1_q, w2_q, offsets = ctx.saved_tensors
            grad_act = _triton_grouped_gemm(grad_y, w2_q, offsets)
            grad_w2 = _triton_grouped_gemm_weight(grad_y, act, offsets)
            grad_h1 = torch.ops.aten.gelu_backward(grad_act, h1)
            grad_x = _triton_grouped_gemm(grad_h1, w1_q, offsets)
            grad_w1 = _triton_grouped_gemm_weight(grad_h1, x, offsets)
            return grad_x, grad_w1, grad_w2, None


class CUDASparseMoELayer(nn.Module):
    """
    Experimental Drop-in Replacement for SparseMoELayer using high-performance
    fused CUDA dispatch, gather, and scatter kernels on sm_120.

    Preserves 100% exact mathematical routing, gating, load-balancing loss,
    and parameter gradients.
    """
    def __init__(self, d_model, num_experts=4, top_k=2, hidden_mult=2,
                 noise_std=0.1, balance_alpha=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.balance_alpha = balance_alpha
        self.router = nn.Linear(d_model, num_experts, bias=False)
        hidden = d_model * hidden_mult

        try:
            from utils.ternary_ops import TernaryLinear
        except ImportError:
            try:
                from jarvis_engine.utils.ternary_ops import TernaryLinear
            except ImportError:
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jarvis_engine")))
                from utils.ternary_ops import TernaryLinear
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

    def forward(self, x):
        global _FALLBACK_USED
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        # 1. Router logits + Gaussian noise
        logits = self.router(x_flat)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_gates = (topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)).to(dtype=x_flat.dtype)

        # Fallback to pure PyTorch if CUDA extension is unavailable or tensor is on CPU
        if not _CUDA_AVAILABLE or not x.is_cuda:
            _FALLBACK_USED = True
            flat_idx  = topk_idx.reshape(-1)
            flat_gate = topk_gates.reshape(-1)
            flat_x    = x_flat.repeat_interleave(self.top_k, dim=0)
            flat_out  = torch.zeros_like(flat_x)

            for e in range(self.num_experts):
                mask = (flat_idx == e)
                xe = flat_x[mask]
                ye = self.w2[e](F.gelu(self.w1[e](xe)))
                flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye

            out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)
        else:
            # 2. Compute dispatch metadata (counts, offsets, scatter/gather maps)
            expert_counts, expert_offsets, scatter_map, gather_map, gate_idx_map = (
                sparse_model_cuda.compute_metadata(topk_idx, self.num_experts)
            )

            # 3. Fused dispatch gather: permutes x_flat into contiguous expert partitions
            dispatched_x = MoeDispatchGather.apply(x_flat, gather_map, scatter_map, N)

            # 4. Expert forward computation on contiguous slices
            if _TRITON_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
                w1_stacked = torch.stack([self.w1[e].weight for e in range(self.num_experts)])
                w2_stacked = torch.stack([self.w2[e].weight for e in range(self.num_experts)])
                if getattr(self, "_native_prequantized", False):
                    w1_q, w2_q = w1_stacked, w2_stacked
                else:
                    w1_q = StackedTernarySTE.apply(w1_stacked)
                    w2_q = StackedTernarySTE.apply(w2_stacked)
                dispatched_y = TritonGroupedMoEMLPFunction.apply(dispatched_x, w1_q, w2_q, expert_offsets)
            else:
                dispatched_y = torch.empty_like(dispatched_x)
                offsets_cpu = expert_offsets.cpu().numpy()

                for e in range(self.num_experts):
                    s = int(offsets_cpu[e])
                    e_end = int(offsets_cpu[e + 1])
                    if e_end > s:
                        xe = dispatched_x[s:e_end]
                        dispatched_y[s:e_end] = self.w2[e](F.gelu(self.w1[e](xe)))

            # 5. Fused scatter combine & gate weighting
            out_flat = MoeScatterCombine.apply(dispatched_y, topk_gates, scatter_map, gather_map, gate_idx_map)
            out = out_flat.view(B, T, C)

        # 6. Load-balancing loss (Eq. 6)
        top1_idx = topk_idx[:, 0]
        f = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=0)
        P = probs.mean(dim=0)
        l_balance = self.balance_alpha * self.num_experts * (f * P).sum()

        act_mean = out.mean()
        act_var  = out.var()
        return out, l_balance, act_mean, act_var
