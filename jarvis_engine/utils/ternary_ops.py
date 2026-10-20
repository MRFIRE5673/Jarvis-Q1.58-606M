import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl
    _TRITON_STE_AVAILABLE = True
except ImportError:
    _TRITON_STE_AVAILABLE = False

if _TRITON_STE_AVAILABLE:
    @triton.jit
    def _ternary_quantize_fwd_kernel(
        W_ptr, Wq_ptr, Alpha_ptr,
        N_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements
        
        alpha = tl.load(Alpha_ptr)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w_norm = w / alpha
        w_clamped = tl.maximum(tl.minimum(w_norm, 1.0), -1.0)
        w_round = tl.extra.cuda.libdevice.nearbyint(w_clamped)
        w_q = w_round * alpha
        
        tl.store(Wq_ptr + offsets, w_q.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _ternary_quantize_bwd_kernel(
        GradOut_ptr, W_ptr, Alpha_ptr, GradW_ptr,
        N_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements
        
        go = tl.load(GradOut_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        alpha = tl.load(Alpha_ptr).to(tl.float32)
        is_valid = tl.abs(w) <= alpha
        gw = tl.where(is_valid, go, 0.0)
        
        tl.store(GradW_ptr + offsets, gw.to(GradW_ptr.dtype.element_ty), mask=mask)

    class FusedTernaryQuantizeSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w):
            if not w.is_cuda:
                alpha = w.abs().mean().clamp(min=1e-8)
                w_norm = w / alpha
                w_q = torch.round(torch.clamp(w_norm, -1.0, 1.0))
                ctx.save_for_backward(w, alpha)
                return w_q * alpha

            alpha = w.abs().mean().clamp(min=1e-8)
            wq = torch.empty_like(w)
            N = w.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            _ternary_quantize_fwd_kernel[grid](
                w, wq, alpha,
                N, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
            ctx.save_for_backward(w, alpha)
            return wq

        @staticmethod
        def backward(ctx, grad_output):
            w, alpha = ctx.saved_tensors
            if not w.is_cuda:
                mask = (w.abs() <= alpha).to(dtype=grad_output.dtype)
                return grad_output * mask

            grad_w = torch.empty_like(w)
            N = w.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            _ternary_quantize_bwd_kernel[grid](
                grad_output, w, alpha, grad_w,
                N, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
            return grad_w


if _TRITON_STE_AVAILABLE:
    TernaryQuantizeSTE = FusedTernaryQuantizeSTE
else:
    class TernaryQuantizeSTE(torch.autograd.Function):
        """
        Ternary weight quantization with AbsMean scaling.

        Forward (Eq. 3):
            α   = mean(|W_FP32|)          — AbsMean scale (BitNet convention)
            W̃  = round(clamp(W / α, -1, 1)) · α   — maps to {-1, 0, +1} in original scale

        Backward (Eq. 5 — Straight-Through Estimator):
            ∂L/∂W_FP32 ≈ ∂L/∂W̃ · 1{|W / α| ≤ 1}
        """

        @staticmethod
        def forward(ctx, w):
            alpha = w.abs().mean().clamp(min=1e-8)      # AbsMean scale (BitNet convention)
            w_norm = w / alpha                           # normalise to ≈ unit scale
            w_q = torch.round(torch.clamp(w_norm, -1.0, 1.0))
            ctx.save_for_backward(w, alpha)              # save master weights and AbsMean threshold
            return w_q * alpha                           # rescale back to original magnitude

        @staticmethod
        def backward(ctx, grad_output):
            w, alpha = ctx.saved_tensors
            mask = (w.abs() <= alpha).to(dtype=grad_output.dtype)
            return grad_output * mask


class TernaryLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weights via STE."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w_q = self.weight if getattr(self, "_native_prequantized", False) else TernaryQuantizeSTE.apply(self.weight)
        return F.linear(x, w_q, self.bias)


class FusedQKVSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        # weight: (3 * d_model, d_model) -> 3 projections: Q, K, V
        alpha = weight.abs().mean().clamp(min=1e-8)
        ctx.save_for_backward(weight, alpha)
        w_norm = (weight / alpha).clamp(-1.0, 1.0).round()
        return w_norm * alpha

    @staticmethod
    def backward(ctx, grad_w_q):
        weight, alpha = ctx.saved_tensors
        mask = (weight.abs() <= alpha).to(grad_w_q.dtype)
        return grad_w_q * mask


class FusedQKVLinear(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        w_q = self.weight if getattr(self, "_native_prequantized", False) else FusedQKVSTE.apply(self.weight)
        return F.linear(x, w_q)


# Optimized batched MoE computation using grouped GEMM
class BatchedTernaryExpert(nn.Module):
    """Batched ternary expert computation for MoE - processes all experts in parallel."""
    def __init__(self, d_model, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        # Stacked weights for batched GEMM: (E, hidden, d_model) and (E, d_model, hidden)
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        nn.init.kaiming_uniform_(self.w1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.w2, a=5 ** 0.5)
        
    def forward(self, x, expert_indices, gates):
        """
        x: (N*K, C) - input tokens
        expert_indices: (N*K,) - which expert each token goes to
        gates: (N*K,) - gate values
        """
        # For now, use the existing per-expert loop but with optimized batching
        # A full grouped GEMM implementation would require custom CUDA kernels
        raise NotImplementedError("Use SparseMoELayer for now - this is a placeholder for future optimization")
