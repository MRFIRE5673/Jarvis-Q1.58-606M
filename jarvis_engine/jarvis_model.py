"""
Jarvis Neuromorphic Transformer - fixed to match research paper equations.

Bugs fixed vs. original:
  1. AssociativeLinearAttention: replaced chunked intra-chunk softmax attention
     with the correct per-token O(N) linear recurrence (Eq. 1-2, Algo 1 L4-6).
  2. SparseMoELayer: added Gaussian noise injection before Top-K (Algo 1 L11).
  3. SparseMoELayer: fixed load-balance loss to use f_i * P_i (Eq. 6) not P_i^2.
  4. Reflective Penalty (Eq. 7) implemented and wired into the loss.
  5. LiquidStateFusion: changed to EMA H_t = alpha*H_{t-1} + (1-alpha)*M_t
     with alpha computed dynamically from expert output variance (Algo 1 L15).
  6. LiquidStateFusion: membrane state now persists across the full sequence.
"""
import os, sys, glob, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from utils.ternary_ops import TernaryLinear, FusedQKVSTE, FusedQKVLinear


try:
    import triton
    import triton.language as tl
    _TRITON_RMSNORM_AVAILABLE = True
except ImportError:
    _TRITON_RMSNORM_AVAILABLE = False

if _TRITON_RMSNORM_AVAILABLE:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr, Y_ptr, W_ptr, Rsqrt_ptr,
        stride_x_row, stride_y_row,
        N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x_ptrs = X_ptr + row_idx * stride_x_row + cols
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        var = tl.sum(x * x, axis=0) / N
        rsqrt = 1.0 / tl.sqrt(var + eps)
        tl.store(Rsqrt_ptr + row_idx, rsqrt)
        
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rsqrt * w
        
        y_ptrs = Y_ptr + row_idx * stride_y_row + cols
        tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _rmsnorm_bwd_dx_kernel(
        GradY_ptr, X_ptr, W_ptr, Rsqrt_ptr, GradX_ptr,
        stride_gy_row, stride_x_row, stride_gx_row,
        N: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        gy_ptrs = GradY_ptr + row_idx * stride_gy_row + cols
        x_ptrs = X_ptr + row_idx * stride_x_row + cols
        
        gy = tl.load(gy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(Rsqrt_ptr + row_idx).to(tl.float32)
        
        x_w_gy = x * w * gy
        sum_x_w_gy = tl.sum(x_w_gy, axis=0)
        c = (rsqrt * rsqrt * rsqrt / N) * sum_x_w_gy
        gx = rsqrt * w * gy - c * x
        
        gx_ptrs = GradX_ptr + row_idx * stride_gx_row + cols
        tl.store(gx_ptrs, gx.to(tl.bfloat16), mask=mask)

    class FusedRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps=1e-6):
            orig_shape = x.shape
            N = orig_shape[-1]
            x_2d = x.contiguous().view(-1, N)
            M = x_2d.shape[0]
            y_2d = torch.empty_like(x_2d)
            rsqrt = torch.empty(M, dtype=torch.float32, device=x.device)
            
            BLOCK_SIZE = triton.next_power_of_2(N)
            grid = (M,)
            _rmsnorm_fwd_kernel[grid](
                x_2d, y_2d, weight, rsqrt,
                x_2d.stride(0), y_2d.stride(0),
                N=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
            ctx.save_for_backward(x_2d, weight, rsqrt)
            ctx.eps = eps
            ctx.orig_shape = orig_shape
            return y_2d.view(orig_shape)

        @staticmethod
        def backward(ctx, grad_y):
            x_2d, weight, rsqrt = ctx.saved_tensors
            M, N = x_2d.shape
            grad_y_2d = grad_y.contiguous().view(-1, N)
            grad_x_2d = torch.empty_like(x_2d)
            
            BLOCK_SIZE = triton.next_power_of_2(N)
            grid = (M,)
            _rmsnorm_bwd_dx_kernel[grid](
                grad_y_2d, x_2d, weight, rsqrt, grad_x_2d,
                grad_y_2d.stride(0), x_2d.stride(0), grad_x_2d.stride(0),
                N=N, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
            grad_weight = (grad_y_2d * (x_2d * rsqrt.unsqueeze(1))).sum(dim=0).to(weight.dtype)
            return grad_x_2d.view(ctx.orig_shape), grad_weight, None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if x.is_cuda and _TRITON_RMSNORM_AVAILABLE:
            return FusedRMSNormFunction.apply(x, self.weight, self.eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE, Su et al. 2021).
    Encodes relative position directly into Q and K in AssociativeLinearAttention,
    eliminating the need for fixed-size absolute pos_emb tables and enabling
    unbounded / infinite sequence lengths.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self._build_cache(max_seq_len, device=inv_freq.device)

    def _build_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = max(seq_len, 256)
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (max_seq_len_cached, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)           # (max_seq_len_cached, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        # q, k: (B, H, T, D)
        T = q.shape[2]
        end_pos = start_pos + T
        if end_pos > self.max_seq_len_cached or self.cos_cached.device != q.device:
            new_len = max(end_pos, self.max_seq_len_cached * 2)
            self._build_cache(new_len, device=q.device)

        cos = self.cos_cached[start_pos:end_pos].to(dtype=q.dtype)[None, None, :, :]  # (1, 1, T, D)
        sin = self.sin_cached[start_pos:end_pos].to(dtype=q.dtype)[None, None, :, :]  # (1, 1, T, D)
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Vectorized Chunked O(N) Associative Linear Attention (Eq. 1-2, Algo 1 L4-6)
# with RoPE (Rotary Position Embeddings)
#
# Paper recurrence (per token):
#   S_t = γ * S_{t-1} + v_t ⊗ k_t^T
#   z_t = S_t @ q_t
#
# Optimization: instead of a T-step Python loop (256 serial kernel launches),
# we process CHUNK_SIZE=64-token chunks with fully vectorized matmuls (4 iters).
# Each chunk decomposes z_i (local index i within chunk) into:
#   Cross-chunk : γ^{i+1} * (S_prev @ q_i)               from carried state
#   Intra-chunk : Σ_{j≤i} γ^{i-j} * (k_j · q_i) * v_j   decayed attn within chunk
# State update  : S_new = γ^c * S_prev + Σ_j γ^{c-1-j} * v_j ⊗ k_j^T
# ---------------------------------------------------------------------------
class AssociativeLinearAttention(nn.Module):
    CHUNK_SIZE = 64

    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Fused QKV projection - single ternary quantization for all three
        self.qkv_proj = FusedQKVLinear(d_model)
        self.out_proj = TernaryLinear(d_model, d_model)
        # γ: per-head learnable decay, init sigmoid(2.94) ≈ 0.95
        self.gamma_raw = nn.Parameter(torch.full((n_heads,), 2.94))
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Pre-cache chunk buffers — allocated ONCE at init, reused every forward.
        # Eliminates torch.arange + outer-product diff + clamp + comparison per step.
        cs = self.CHUNK_SIZE
        i = torch.arange(cs, dtype=torch.float32)           # (cs,)
        diff = i.unsqueeze(1) - i.unsqueeze(0)              # (cs, cs)
        self.register_buffer('_diff_clamp', diff.clamp(min=0))          # (cs, cs)
        self.register_buffer('_causal',     (diff >= 0).float())         # (cs, cs)
        self.register_buffer('_i_idx',      i)                           # (cs,)
        self.register_buffer('_i_idx_p1',   i + 1)                       # (cs,) i+1
        self.register_buffer('_c_m1_m_i',   (cs - 1) - i)               # (cs,) c-1-i

    def forward(
        self, x: torch.Tensor, start_pos: int = 0,
        s_prev: torch.Tensor = None
    ):
        """Returns (out, s_new).

        s_prev: carried associative state (B, H, D, D) or None.
                When not None (inference/persist_state mode), the full-context
                chunk scan is bypassed — each token is processed in O(1) via
                the paper Eq.1-2 recurrence:  S_t = γ*S_{t-1} + v_t⊗k_t^T
                                               z_t = S_t @ q_t
        """
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # Fused QKV projection - single matmul + ternary quantization
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)

        # ELU+1 keeps feature values positive (Katharopoulos 2020)
        q = (F.elu(q) + 1.0) / math.sqrt(D)
        k = F.elu(k) + 1.0

        # RoPE (Su et al. 2021): inject relative position information
        q, k = self.rotary(q, k, start_pos=start_pos)

        log_g = torch.log(torch.sigmoid(self.gamma_raw))  # (H,) float32

        # ------------------------------------------------------------------
        # O(1) INFERENCE PATH  (s_prev is provided — single-token generation)
        # Processes T tokens sequentially using the pure recurrence, carrying
        # S_t across tokens.  No chunk overhead; each step is one einsum.
        # ------------------------------------------------------------------
        if s_prev is not None:
            gamma = torch.sigmoid(self.gamma_raw).to(dtype=x.dtype)  # (H,)
            state = s_prev
            outputs = []
            for t in range(T):
                q_t = q[:, :, t, :]   # (B, H, D)
                k_t = k[:, :, t, :]   # (B, H, D)
                v_t = v[:, :, t, :]   # (B, H, D)
                # S_t = γ * S_{t-1} + v_t ⊗ k_t^T   (paper Eq. 1)
                kv  = torch.einsum('bhd,bhe->bhde', v_t, k_t)          # (B,H,D,D)
                state = gamma.view(1, H, 1, 1) * state + kv
                # z_t = S_t @ q_t                    (paper Eq. 2)
                z_t = torch.einsum('bhde,bhe->bhd', state, q_t)         # (B,H,D)
                outputs.append(z_t.unsqueeze(2))                         # (B,H,1,D)
            out_h = torch.cat(outputs, dim=2)                            # (B,H,T,D)
            out = out_h.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out), state

        # ------------------------------------------------------------------
        # TRAINING / PREFILL PATH  (chunked parallel scan, s_prev=None)
        # ------------------------------------------------------------------
        cs = self.CHUNK_SIZE
        dc    = self._diff_clamp   # (cs, cs) float32
        ca    = self._causal       # (cs, cs) float32
        ip1   = self._i_idx_p1     # (cs,)    float32
        cm1mi = self._c_m1_m_i     # (cs,)    float32

        decay_mat   = (torch.exp(log_g.view(H,1,1) * dc) * ca).to(dtype=x.dtype)   # (H,cs,cs)
        gamma_cross = torch.exp(log_g.view(H,1) * ip1).to(dtype=x.dtype)          # (H,cs)
        gw          = torch.exp(log_g.view(H,1) * cm1mi).to(dtype=x.dtype)        # (H,cs)
        gamma_c     = torch.exp(log_g * cs).view(1,H,1,1).to(dtype=x.dtype)       # (1,H,1,1)

        state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        for start in range(0, T, cs):
            end   = min(start + cs, T)
            c     = end - start
            q_c   = q[:, :, start:end, :]
            k_c   = k[:, :, start:end, :]
            v_c   = v[:, :, start:end, :]

            if c == cs:
                dm       = decay_mat
                gc_cross = gamma_cross
                gw_c     = gw
                gc_state = gamma_c
            else:
                i_c      = self._i_idx[:c]
                diff_c   = (i_c.unsqueeze(1) - i_c.unsqueeze(0)).clamp(min=0)
                causal_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0) >= 0).float()
                dm       = (torch.exp(log_g.view(H,1,1) * diff_c) * causal_c).to(dtype=x.dtype)
                gc_cross = torch.exp(log_g.view(H,1) * (i_c + 1)).to(dtype=x.dtype)
                gw_c     = torch.exp(log_g.view(H,1) * (c - 1 - i_c)).to(dtype=x.dtype)
                gc_state = torch.exp(log_g * c).view(1,H,1,1).to(dtype=x.dtype)

            raw       = torch.einsum('bhid,bhjd->bhij', q_c, k_c)
            scores    = raw * dm.unsqueeze(0)
            intra_out = torch.einsum('bhij,bhjd->bhid', scores, v_c)

            raw_cross = torch.einsum('bhde,bhie->bhid', state, q_c)
            cross_out = raw_cross * gc_cross.unsqueeze(0).unsqueeze(-1)

            outputs.append(intra_out + cross_out)

            v_w       = v_c * gw_c.unsqueeze(0).unsqueeze(-1)
            chunk_upd = torch.einsum('bhid,bhie->bhde', v_w, k_c)
            state     = gc_state * state + chunk_upd

        out = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), state


# ---------------------------------------------------------------------------
# Bug 2 FIX: Gaussian noise added to router logits (Algo 1 L11)
# Bug 3 FIX: Load-balance loss uses f_i * P_i (Eq. 6), not P_i^2
# ---------------------------------------------------------------------------
class SparseMoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=2, hidden_mult=2,
                 noise_std=0.1, balance_alpha=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std        # σ for Gaussian routing noise
        self.balance_alpha = balance_alpha  # α in Eq. 6
        self.router = nn.Linear(d_model, num_experts, bias=False)
        hidden = d_model * hidden_mult
        # Paper Section 3.2: ALL Feed-Forward pathways are ternary
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        # Router logits + Gaussian noise (Algo 1 L11: G(A_spike) + N(0, σ²))
        logits = self.router(x_flat)  # (N, E)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)                        # (N, E)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)   # (N, top_k)
        topk_gates = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Vectorized MoE: batched expert computation
        # flat_idx/flat_gate: (N*top_k,)  flat_x: (N*top_k, C)
        flat_idx  = topk_idx.reshape(-1)                         # (N*K,)
        flat_gate = topk_gates.reshape(-1)                       # (N*K,)
        flat_x    = x_flat.repeat_interleave(self.top_k, dim=0)  # (N*K, C)
        flat_out  = torch.zeros_like(flat_x)                     # (N*K, C)

        # Batch experts together using grouped GEMM-like pattern
        # Each expert processes its assigned tokens in parallel
        for e in range(self.num_experts):
            mask = (flat_idx == e)
            if mask.any():
                xe = flat_x[mask]
                ye = self.w2[e](F.gelu(self.w1[e](xe)))
                flat_out[mask] = flat_gate[mask].unsqueeze(-1) * ye

        # Sum both top-k contributions per original token
        out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)

        # Load-balance loss: α * N_experts * Σ f_i * P_i  (Eq. 6)
        top1_idx = topk_idx[:, 0]
        f = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=0)
        P = probs.mean(dim=0)
        l_balance = self.balance_alpha * self.num_experts * (f * P).sum()

        act_mean = out.mean()
        act_var  = out.var()
        return out, l_balance, act_mean, act_var


# ---------------------------------------------------------------------------
# Bug 5 & 6 FIX: Liquid State Fusion with correct EMA + dynamic α + state persistence
#
# Paper (Algo 1 L15):  H_t = α * H_{t-1} + (1 - α) * M_t
# α is dynamically derived from the variance of the expert output subset.
# The membrane potential H must persist across ALL tokens in the sequence.
# ---------------------------------------------------------------------------
if _TRITON_RMSNORM_AVAILABLE:
    @triton.jit
    def _streaming_lsf_fwd_kernel(
        X_ptr, H_ptr, H_last_ptr, H0_ptr, Alpha_ptr,
        stride_xb, stride_xt, stride_xd,
        stride_hb, stride_ht, stride_hd,
        stride_lb, stride_ld,
        stride_0b, stride_0d,
        B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
        has_h0: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        pid_d = tl.program_id(0)
        pid_b = tl.program_id(1)
        
        alpha = tl.load(Alpha_ptr).to(tl.float32)
        one_minus_alpha = 1.0 - alpha
        
        col_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = col_offsets < D
        
        if has_h0:
            h0_ptrs = H0_ptr + pid_b * stride_0b + col_offsets * stride_0d
            h = tl.load(h0_ptrs, mask=mask, other=0.0).to(tl.float32)
        else:
            h = tl.zeros((BLOCK_D,), dtype=tl.float32)
            
        x_base = X_ptr + pid_b * stride_xb + col_offsets * stride_xd
        h_base = H_ptr + pid_b * stride_hb + col_offsets * stride_hd
        
        for t in range(0, T):
            x_ptrs = x_base + t * stride_xt
            x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            h = alpha * h + one_minus_alpha * x
            h_ptrs = h_base + t * stride_ht
            tl.store(h_ptrs, h.to(tl.bfloat16), mask=mask)
            
        h_last_ptrs = H_last_ptr + pid_b * stride_lb + col_offsets * stride_ld
        tl.store(h_last_ptrs, h.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _streaming_lsf_bwd_kernel(
        GradH_ptr, GradHLast_ptr, X_ptr, H_ptr, H0_ptr, Alpha_ptr,
        GradX_ptr, GradAlpha_block_ptr, GradH0_ptr,
        stride_ghb, stride_ght, stride_ghd,
        stride_xb, stride_xt, stride_xd,
        stride_hb, stride_ht, stride_hd,
        stride_gxb, stride_gxt, stride_gxd,
        stride_0b, stride_0d,
        B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
        has_h0: tl.constexpr, has_gh_last: tl.constexpr,
        BLOCK_D: tl.constexpr
    ):
        pid_d = tl.program_id(0)
        pid_b = tl.program_id(1)
        
        alpha = tl.load(Alpha_ptr).to(tl.float32)
        one_minus_alpha = 1.0 - alpha
        
        col_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = col_offsets < D
        
        lambda_val = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if has_gh_last:
            gh_last_ptrs = GradHLast_ptr + pid_b * D + col_offsets
            lambda_val += tl.load(gh_last_ptrs, mask=mask, other=0.0).to(tl.float32)
            
        d_alpha_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        gh_base = GradH_ptr + pid_b * stride_ghb + col_offsets * stride_ghd
        x_base = X_ptr + pid_b * stride_xb + col_offsets * stride_xd
        h_base = H_ptr + pid_b * stride_hb + col_offsets * stride_hd
        gx_base = GradX_ptr + pid_b * stride_gxb + col_offsets * stride_gxd
        
        for t in range(T - 1, -1, -1):
            gh_ptrs = gh_base + t * stride_ght
            gh = tl.load(gh_ptrs, mask=mask, other=0.0).to(tl.float32)
            lambda_val = lambda_val + gh
            
            gx = one_minus_alpha * lambda_val
            gx_ptrs = gx_base + t * stride_gxt
            tl.store(gx_ptrs, gx.to(tl.bfloat16), mask=mask)
            
            if t > 0:
                h_prev_ptrs = h_base + (t - 1) * stride_ht
                h_prev = tl.load(h_prev_ptrs, mask=mask, other=0.0).to(tl.float32)
            else:
                if has_h0:
                    h0_ptrs = H0_ptr + pid_b * stride_0b + col_offsets * stride_0d
                    h_prev = tl.load(h0_ptrs, mask=mask, other=0.0).to(tl.float32)
                else:
                    h_prev = tl.zeros((BLOCK_D,), dtype=tl.float32)
                    
            x_ptrs = x_base + t * stride_xt
            x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            d_alpha_acc += lambda_val * (h_prev - x)
            lambda_val = lambda_val * alpha
            
        sum_d_alpha = tl.sum(d_alpha_acc, axis=0)
        num_blocks_d = tl.cdiv(D, BLOCK_D)
        block_id = pid_b * num_blocks_d + pid_d
        tl.store(GradAlpha_block_ptr + block_id, sum_d_alpha)
        
        if has_h0:
            gh0_ptrs = GradH0_ptr + pid_b * stride_0b + col_offsets * stride_0d
            tl.store(gh0_ptrs, lambda_val.to(tl.bfloat16), mask=mask)

    class TritonStreamingLSFFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha, h0=None):
            B, T, D = x.shape
            H = torch.empty_like(x)
            H_last = torch.empty(B, D, dtype=x.dtype, device=x.device)
            
            BLOCK_D = 64
            num_blocks_d = triton.cdiv(D, BLOCK_D)
            grid = (num_blocks_d, B)
            
            has_h0 = h0 is not None
            h0_ptr = h0 if has_h0 else x
            stride_0b = h0.stride(0) if has_h0 else 0
            stride_0d = h0.stride(1) if has_h0 else 0
            
            _streaming_lsf_fwd_kernel[grid](
                x, H, H_last, h0_ptr, alpha,
                x.stride(0), x.stride(1), x.stride(2),
                H.stride(0), H.stride(1), H.stride(2),
                H_last.stride(0), H_last.stride(1),
                stride_0b, stride_0d,
                B=B, T=T, D=D,
                has_h0=has_h0, BLOCK_D=BLOCK_D,
                num_warps=2
            )
            ctx.save_for_backward(x, H, h0, alpha)
            ctx.has_h0 = has_h0
            return H, H_last

        @staticmethod
        def backward(ctx, grad_H, grad_H_last):
            x, H, h0, alpha = ctx.saved_tensors
            B, T, D = x.shape
            has_h0 = ctx.has_h0
            has_gh_last = grad_H_last is not None
            
            GradX = torch.empty_like(x)
            GradH0 = torch.empty_like(h0) if has_h0 else None
            
            BLOCK_D = 64
            num_blocks_d = triton.cdiv(D, BLOCK_D)
            grid = (num_blocks_d, B)
            
            GradAlpha_blocks = torch.empty(B * num_blocks_d, dtype=torch.float32, device=x.device)
            h0_ptr = h0 if has_h0 else x
            gh0_ptr = GradH0 if has_h0 else x
            stride_0b = h0.stride(0) if has_h0 else 0
            stride_0d = h0.stride(1) if has_h0 else 0
            gh_last_ptr = grad_H_last if has_gh_last else grad_H
            
            _streaming_lsf_bwd_kernel[grid](
                grad_H, gh_last_ptr, x, H, h0_ptr, alpha,
                GradX, GradAlpha_blocks, gh0_ptr,
                grad_H.stride(0), grad_H.stride(1), grad_H.stride(2),
                x.stride(0), x.stride(1), x.stride(2),
                H.stride(0), H.stride(1), H.stride(2),
                GradX.stride(0), GradX.stride(1), GradX.stride(2),
                stride_0b, stride_0d,
                B=B, T=T, D=D,
                has_h0=has_h0, has_gh_last=has_gh_last,
                BLOCK_D=BLOCK_D,
                num_warps=2
            )
            grad_alpha = GradAlpha_blocks.sum().to(alpha.dtype)
            return GradX, grad_alpha, GradH0


# ---------------------------------------------------------------------------
# LiquidStateFusion (Algo 1 L15, Eq. 8)
# ---------------------------------------------------------------------------
class LiquidStateFusion(nn.Module):
    """
    Leaky Integrate-and-Fire membrane state recurrence (Algo 1 L15).

    Mathematical recurrence:
      H_t = α · H_{t-1} + (1 − α) · M_t,  t ∈ [0, T-1]

    Executed via high-speed streaming Triton recurrence on GPU, with
    parallel causal scan fallback on CPU / unsupported backends.
    """
    def __init__(self, d_model, alpha_min=0.1, alpha_max=0.99):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.var_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, act_var, h_prev=None):
        """
        Args:
            x:       MoE output          (B, T, C)
            act_var: scalar MoE variance  (from SparseMoELayer)
            h_prev:  last-token membrane  (B, C) or None — carry from previous sequence
        Returns:
            h_out:   per-token states     (B, T, C)
            h_last:  last-token state     (B, C)  — persist for next sequence if desired
        """
        B, T, C = x.shape
        # Initialise membrane to zero if no prior state (start of sequence)
        if h_prev is None:
            h_prev = x.new_zeros(B, C)

        # Dynamic α from variance (Eq. 8 / Algo 1 L15)
        alpha_raw = torch.sigmoid(-self.var_scale * act_var)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw

        # Fast path: High-throughput Triton streaming recurrence
        if x.is_cuda and _TRITON_RMSNORM_AVAILABLE:
            h_out, h_last = TritonStreamingLSFFunction.apply(x, alpha, h_prev)
            return h_out, h_last

        # Fallback: Parallel causal scan
        t_idx = torch.arange(T, device=x.device, dtype=torch.float32)
        diff = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0)).clamp(min=0)
        causal = (t_idx.unsqueeze(1) - t_idx.unsqueeze(0) >= 0).to(dtype=x.dtype)

        log_a = torch.log(alpha)
        decay_mat = (torch.exp(log_a * diff) * causal).to(dtype=x.dtype)
        conv_out = (1.0 - alpha) * torch.matmul(decay_mat, x)

        carry_weights = torch.exp(log_a * (t_idx + 1)).view(1, T, 1).to(dtype=x.dtype)
        carry_out = carry_weights * h_prev.unsqueeze(1)

        h_out = conv_out + carry_out
        h_last = h_out[:, -1, :]
        return h_out, h_last


# ---------------------------------------------------------------------------
# Bug 4 FIX: Reflective Penalty (Eq. 7)
#
#   L_reflect = λ * [(µ_t - µ_batch)² + max(0, σ² - τ_max)]
#
# µ_t  = exponentially tracked running mean (updated each step)
# µ_batch = current batch activation mean
# σ²   = current batch activation variance
# τ_max = maximum allowable variance threshold
# ---------------------------------------------------------------------------
class ReflectivePenalty(nn.Module):
    """
    Tracks a running mean µ_t and penalises deviations from it and
    excess variance in activations (Eq. 7).
    """
    def __init__(self, lam=0.01, tau_max=1.0, ema_decay=0.99):
        super().__init__()
        self.lam = lam
        self.tau_max = tau_max
        self.ema_decay = ema_decay
        # Running mean µ_t — not a parameter, just a buffer
        self.register_buffer('mu_t', torch.tensor(0.0))

    def forward(self, act_mean, act_var):
        mu_batch = act_mean
        # L_reflect = λ * [(µ_t - µ_batch)² + max(0, σ² - τ_max)]
        mean_penalty = (self.mu_t.detach() - mu_batch) ** 2
        var_penalty = F.relu(act_var - self.tau_max)
        l_reflect = self.lam * (mean_penalty + var_penalty)

        # Update running mean (EMA) in-place for CUDA Graph memory stability
        if self.training:
            self.mu_t.copy_(self.ema_decay * self.mu_t + (1.0 - self.ema_decay) * mu_batch.detach())

        return l_reflect


# ---------------------------------------------------------------------------
# Native CUDA Acceleration with Automatic Pure-PyTorch Fallback
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _ext_dir in [
    os.path.join(_WORKSPACE_ROOT, "associative_attention_cuda"),
    os.path.join(_WORKSPACE_ROOT, "sparse_model_cuda"),
]:
    if os.path.isdir(_ext_dir) and _ext_dir not in sys.path:
        sys.path.insert(0, _ext_dir)

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

_HAS_CUDA_ATTN = False
_HAS_CUDA_MOE = False

try:
    from associative_attention import CUDAAssociativeLinearAttention
    _HAS_CUDA_ATTN = True
except Exception:
    CUDAAssociativeLinearAttention = None

try:
    from sparse_model import CUDASparseMoELayer
    _HAS_CUDA_MOE = True
except Exception:
    CUDASparseMoELayer = None


class JarvisBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts=4, top_k=2, max_seq_len=2048,
                 use_cuda_attn: bool = True, use_cuda_moe: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if use_cuda_attn and _HAS_CUDA_ATTN:
            self.attn = CUDAAssociativeLinearAttention(d_model, n_heads, max_seq_len=max_seq_len)
        else:
            self.attn = AssociativeLinearAttention(d_model, n_heads, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(d_model)
        if use_cuda_moe and _HAS_CUDA_MOE:
            self.moe = CUDASparseMoELayer(d_model, num_experts=num_experts, top_k=top_k)
        else:
            self.moe = SparseMoELayer(d_model, num_experts=num_experts, top_k=top_k)
        self.liquid = LiquidStateFusion(d_model)
        self.reflect = ReflectivePenalty()

    def forward(self, x, h_prev=None, start_pos: int = 0, s_prev=None):
        """
        Args:
            x:         input hidden states  (B, T, C)
            h_prev:    last-token membrane  (B, C) or None  [carry from previous call]
            start_pos: token position offset for RoPE
            s_prev:    associative attention state (B, H, D, D) or None.
                       When provided (inference), enables O(1) per-token attention.
        Returns:
            x:       updated hidden states  (B, T, C)
            h_last:  last-token membrane    (B, C)
            s_new:   updated attention state (B, H, D, D)
            l_balance, l_reflect: aux losses
        """
        # Step 1: Associative Attention (Eq. 1-2) with RoPE
        attn_out, s_new = self.attn(self.norm1(x), start_pos=start_pos, s_prev=s_prev)
        x = x + attn_out

        # Step 2: Sparse MoE (Eq. 6 + Algo 1 L11-13)
        moe_out, l_balance, act_mean, act_var = self.moe(self.norm2(x))

        # Step 3: Liquid State — token-sequential EMA scan (Algo 1 L15 / Eq. 8)
        h_out, h_last = self.liquid(moe_out, act_var, h_prev)
        x = x + h_out

        # Step 4: Reflective Penalty (Eq. 7)
        l_reflect = self.reflect(act_mean, act_var)

        return x, h_last, s_new, l_balance, l_reflect


# ---------------------------------------------------------------------------
# Padded LM Head Autograd Function (Internal 64-Tile Alignment)
# ---------------------------------------------------------------------------
class PaddedLMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, pad_n: int):
        V, K = weight.shape
        M = x.shape[0]
        pad_rows = pad_n - V
        
        zero_w = torch.zeros(pad_rows, K, dtype=weight.dtype, device=weight.device)
        w_pad = torch.cat([weight, zero_w], dim=0)
        
        logits_pad = F.linear(x, w_pad)
        logits = logits_pad[:, :V]
        
        ctx.save_for_backward(x, w_pad)
        ctx.V = V
        ctx.pad_n = pad_n
        return logits

    @staticmethod
    def backward(ctx, grad_logits):
        x, w_pad = ctx.saved_tensors
        V, pad_n = ctx.V, ctx.pad_n
        M = grad_logits.shape[0]
        pad_rows = pad_n - V
        
        zero_g = torch.zeros(M, pad_rows, dtype=grad_logits.dtype, device=grad_logits.device)
        grad_logits_pad = torch.cat([grad_logits, zero_g], dim=1)
        
        grad_x = torch.matmul(grad_logits_pad, w_pad.to(grad_logits_pad.dtype))
        grad_w_pad = torch.matmul(grad_logits_pad.t(), x.to(grad_logits_pad.dtype))
        grad_w = grad_w_pad[:V, :].to(w_pad.dtype)
        
        return grad_x, grad_w, None


class Jarvis(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_layers=24, n_heads=16,
                 num_experts=4, top_k=2, max_seq_len=1024,
                 use_cuda_attn: bool = True, use_cuda_moe: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_cuda_attn = use_cuda_attn and _HAS_CUDA_ATTN
        self.use_cuda_moe = use_cuda_moe and _HAS_CUDA_MOE
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Note: pos_emb removed — replaced by RoPE in AssociativeLinearAttention
        self.blocks = nn.ModuleList([
            JarvisBlock(
                d_model, n_heads, num_experts, top_k, max_seq_len=max_seq_len,
                use_cuda_attn=use_cuda_attn, use_cuda_moe=use_cuda_moe
            ) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Persistent cross-sequence states (inference / persist_state=True only).
        # _h_states: liquid membrane  — shape (B, C) per block
        # _s_states: associative attn — shape (B, H, D, D) per block
        self._h_states: list = [None] * n_layers
        self._s_states: list = [None] * n_layers
        self._token_pos: int = 0

    def get_backend_status(self):
        """Returns the active backend status for attention and MoE across all blocks."""
        attn_cuda_count = sum(1 for b in self.blocks if CUDAAssociativeLinearAttention and isinstance(b.attn, CUDAAssociativeLinearAttention))
        moe_cuda_count = sum(1 for b in self.blocks if CUDASparseMoELayer and isinstance(b.moe, CUDASparseMoELayer))
        total_blocks = len(self.blocks)
        return {
            "has_cuda_attn_extension": _HAS_CUDA_ATTN,
            "has_cuda_moe_extension": _HAS_CUDA_MOE,
            "attn_backend": "cuda" if attn_cuda_count == total_blocks else ("pytorch" if attn_cuda_count == 0 else "mixed"),
            "moe_backend": "cuda" if moe_cuda_count == total_blocks else ("pytorch" if moe_cuda_count == 0 else "mixed"),
            "attn_cuda_blocks": f"{attn_cuda_count}/{total_blocks}",
            "moe_cuda_blocks": f"{moe_cuda_count}/{total_blocks}",
        }

    def reset_state(self):
        """Clear all persisted recurrent states and position (call between unrelated sequences)."""
        self._h_states = [None] * len(self.blocks)
        self._s_states = [None] * len(self.blocks)
        self._token_pos = 0

    def forward(self, idx, targets=None, persist_state=False):
        """
        Args:
            idx:           token indices (B, T)
            targets:       optional targets for loss (B, T)
            persist_state: if True, liquid membrane states and RoPE position offsets
                           are carried across forward calls (infinite-context inference mode).
                           If False (training default), states reset each call.
        """
        B, T = idx.shape
        x = self.tok_emb(idx)

        if persist_state and any(
            state is not None and state.shape[0] != B
            for state in (*self._h_states, *self._s_states)
        ):
            self.reset_state()
        start_pos = self._token_pos if persist_state else 0
        if persist_state:
            self._token_pos += T

        l_bal_total = torch.zeros((), device=idx.device, dtype=torch.float32)
        l_ref_total = torch.zeros((), device=idx.device, dtype=torch.float32)

        # Recurrent state persistence.
        # Training  (persist_state=False): all states reset each call — batches are independent.
        # Inference (persist_state=True):  H (liquid) and S (attention) carried across calls.
        if persist_state:
            h_prevs = self._h_states
            s_prevs = self._s_states
        else:
            h_prevs = [None] * len(self.blocks)
            s_prevs = [None] * len(self.blocks)

        new_h_states = []
        new_s_states = []
        for i, block in enumerate(self.blocks):
            h_prev = h_prevs[i]
            s_prev = s_prevs[i]
            if h_prev is not None and h_prev.shape[0] != B:
                h_prev = None
            if s_prev is not None and s_prev.shape[0] != B:
                s_prev = None

            if self.training:
                # grad_ckpt doesn't accept keyword-only args easily; pass s_prev=None
                # (training never uses persist_state, so s_prev is always None here)
                x, h_last, s_new, l_bal, l_ref = grad_ckpt(
                    block, x, h_prev, start_pos, None, use_reentrant=False
                )
            else:
                x, h_last, s_new, l_bal, l_ref = block(
                    x, h_prev, start_pos=start_pos, s_prev=s_prev
                )

            new_h_states.append(h_last.detach())
            new_s_states.append(s_new.detach())
            l_bal_total = l_bal_total + l_bal
            l_ref_total = l_ref_total + l_ref

        self._h_states = new_h_states
        self._s_states = new_s_states

        x = self.final_norm(x)
        if x.is_cuda and self.lm_head.bias is None and self.lm_head.out_features % 64 != 0:
            pad_n = ((self.lm_head.out_features + 63) // 64) * 64
            orig_shape = x.shape
            x_2d = x.contiguous().view(-1, orig_shape[-1])
            logits_2d = PaddedLMHeadFunction.apply(x_2d, self.lm_head.weight, pad_n)
            logits = logits_2d.view(*orig_shape[:-1], self.lm_head.out_features)
        else:
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce + l_bal_total + l_ref_total

        return logits, loss

    def param_count(self):
        n = sum(p.numel() for p in self.parameters())
        return n, f"{n/1e6:.1f}M parameters"
