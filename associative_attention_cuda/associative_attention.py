# associative_attention_cuda/associative_attention.py
"""
CUDA-Accelerated Associative Linear Attention Backend for Jarvis 606M
=====================================================================
Drop-in replacement for AssociativeLinearAttention with:
1. Fused RoPE + ELU(x)+1 + Q-scaling CUDA kernel (single memory pass)
2. High-throughput batched Tensor Core intra-chunk attention (bmm)
3. Fused recurrent state scan across chunks with FP32 accumulator precision
4. Fallback support with comprehensive diagnostics tracking
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add jarvis_engine to path for RotaryEmbedding and TernaryLinear
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JARVIS_ENGINE_PATH = os.path.join(WORKSPACE_ROOT, "jarvis_engine")
for p in [WORKSPACE_ROOT, JARVIS_ENGINE_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jarvis_model import RotaryEmbedding
from utils.ternary_ops import TernaryLinear

# Diagnostics tracking
_DIAGNOSTICS = {
    "extension_imported": False,
    "fused_rope_elu_executed": False,
    "recurrent_scan_executed": False,
    "fallback_used": False,
}

def get_diagnostics():
    return dict(_DIAGNOSTICS)

def reset_diagnostics():
    global _DIAGNOSTICS
    _DIAGNOSTICS["fused_rope_elu_executed"] = False
    _DIAGNOSTICS["recurrent_scan_executed"] = False
    _DIAGNOSTICS["fallback_used"] = False

# Attempt to load the compiled CUDA extension
_cuda_ext = None
try:
    import associative_attention_cuda as _cuda_ext
    _DIAGNOSTICS["extension_imported"] = True
except ImportError:
    # Try importing from build / local dir
    try:
        cur_dir = os.path.dirname(__file__)
        sys.path.insert(0, cur_dir)
        import associative_attention_cuda as _cuda_ext
        _DIAGNOSTICS["extension_imported"] = True
    except ImportError:
        _DIAGNOSTICS["extension_imported"] = False


# ---------------------------------------------------------------------------
# Autograd Function: Fused RoPE + ELU(x)+1 + Scaling
# ---------------------------------------------------------------------------
class FusedRoPEELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos_tab, sin_tab):
        # q, k: (B, H, T, D)
        # cos_tab, sin_tab: (T, D)
        if _cuda_ext is not None and q.is_cuda and k.is_cuda:
            _DIAGNOSTICS["fused_rope_elu_executed"] = True
            q_rot, k_rot = _cuda_ext.fused_rope_elu_forward(q, k, cos_tab, sin_tab)
        else:
            _DIAGNOSTICS["fallback_used"] = True
            # Reference PyTorch fallback
            head_dim = q.shape[-1]
            q_feat = (F.elu(q) + 1.0) / math.sqrt(head_dim)
            k_feat = F.elu(k) + 1.0
            
            c = cos_tab[None, None, :, :]
            s = sin_tab[None, None, :, :]
            def rotate_half(x):
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)
            q_rot = (q_feat * c) + (rotate_half(q_feat) * s)
            k_rot = (k_feat * c) + (rotate_half(k_feat) * s)

        ctx.save_for_backward(q, k, cos_tab, sin_tab)
        return q_rot, k_rot

    @staticmethod
    def backward(ctx, grad_q_rot, grad_k_rot):
        q, k, cos_tab, sin_tab = ctx.saved_tensors
        if _cuda_ext is not None and grad_q_rot.is_cuda:
            grad_q, grad_k = _cuda_ext.fused_rope_elu_backward(
                grad_q_rot.contiguous(), grad_k_rot.contiguous(),
                q.contiguous(), k.contiguous(),
                cos_tab.contiguous(), sin_tab.contiguous()
            )
        else:
            # Autograd fallback using PyTorch tape
            with torch.enable_grad():
                q_t = q.detach().requires_grad_(True)
                k_t = k.detach().requires_grad_(True)
                head_dim = q.shape[-1]
                q_feat = (F.elu(q_t) + 1.0) / math.sqrt(head_dim)
                k_feat = F.elu(k_t) + 1.0
                c = cos_tab[None, None, :, :]
                s = sin_tab[None, None, :, :]
                def rotate_half(x):
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return torch.cat((-x2, x1), dim=-1)
                qr = (q_feat * c) + (rotate_half(q_feat) * s)
                kr = (k_feat * c) + (rotate_half(k_feat) * s)
                torch.autograd.backward([qr, kr], [grad_q_rot, grad_k_rot])
                grad_q, grad_k = q_t.grad, k_t.grad

        return grad_q, grad_k, None, None


# ---------------------------------------------------------------------------
# Autograd Function: Recurrent Chunk State Scan
# ---------------------------------------------------------------------------
class RecurrentChunkStateScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta_S, gamma_c, h_prev=None):
        # delta_S: (B, H, Nc, D, D)
        # gamma_c: (H,)
        # h_prev: (B, H, D, D) or None
        if _cuda_ext is not None and delta_S.is_cuda:
            _DIAGNOSTICS["recurrent_scan_executed"] = True
            h_prev_t = h_prev if h_prev is not None else torch.empty(0, device=delta_S.device, dtype=delta_S.dtype)
            all_states, h_last = _cuda_ext.recurrent_chunk_state_scan_forward(
                delta_S.contiguous(), gamma_c.contiguous(), h_prev_t
            )
        else:
            _DIAGNOSTICS["fallback_used"] = True
            B, H, Nc, D, _ = delta_S.shape
            states = []
            cur_s = h_prev.clone() if h_prev is not None else torch.zeros(B, H, D, D, device=delta_S.device, dtype=delta_S.dtype)
            gc = gamma_c.view(1, H, 1, 1)
            for k in range(Nc):
                states.append(cur_s)
                cur_s = gc * cur_s + delta_S[:, :, k, :, :]
            all_states = torch.stack(states, dim=2)
            h_last = cur_s

        ctx.save_for_backward(delta_S, all_states, gamma_c)
        ctx.has_h_prev = (h_prev is not None)
        return all_states, h_last

    @staticmethod
    def backward(ctx, grad_all_states, grad_h_last):
        delta_S, all_states, gamma_c = ctx.saved_tensors
        if _cuda_ext is not None and grad_all_states.is_cuda:
            grad_h_last_t = grad_h_last.contiguous() if grad_h_last is not None else torch.empty(0, device=delta_S.device, dtype=delta_S.dtype)
            grad_delta_S, grad_h_prev, grad_gamma_c = _cuda_ext.recurrent_chunk_state_scan_backward(
                grad_all_states.contiguous(),
                grad_h_last_t,
                all_states.contiguous(),
                gamma_c.contiguous(),
                ctx.has_h_prev
            )
        else:
            # PyTorch recurrence fallback
            B, H, Nc, D, _ = delta_S.shape
            grad_delta_S = torch.empty_like(delta_S)
            gc = gamma_c.view(1, H, 1, 1)
            grad_s = grad_h_last.clone() if grad_h_last is not None else torch.zeros(B, H, D, D, device=delta_S.device, dtype=delta_S.dtype)
            grad_gamma_accum = torch.zeros_like(gamma_c, dtype=torch.float32)

            for k in range(Nc - 1, -1, -1):
                grad_delta_S[:, :, k] = grad_s
                grad_gamma_accum += (grad_s.float() * all_states[:, :, k].float()).sum(dim=(0, 2, 3))
                grad_s = grad_all_states[:, :, k] + gc * grad_s

            grad_h_prev = grad_s if ctx.has_h_prev else None
            grad_gamma_c = grad_gamma_accum.to(gamma_c.dtype)

        return grad_delta_S, grad_gamma_c, grad_h_prev if ctx.has_h_prev else None


# ---------------------------------------------------------------------------
# Drop-in Replacement: CUDAAssociativeLinearAttention
# ---------------------------------------------------------------------------
class CUDAAssociativeLinearAttention(nn.Module):
    CHUNK_SIZE = 64

    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.out_proj = TernaryLinear(d_model, d_model)
        self.gamma_raw = nn.Parameter(torch.full((n_heads,), 2.94))
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Pre-cache chunk buffers
        cs = self.CHUNK_SIZE
        i = torch.arange(cs, dtype=torch.float32)
        diff = i.unsqueeze(1) - i.unsqueeze(0)
        self.register_buffer('_diff_clamp', diff.clamp(min=0))
        self.register_buffer('_causal', (diff >= 0).float())
        self.register_buffer('_i_idx_p1', i + 1)
        self.register_buffer('_c_m1_m_i', (cs - 1) - i)
        self.register_buffer('_i_idx', i)

    def forward(
        self, x: torch.Tensor, start_pos: int = 0,
        h_prev: torch.Tensor = None,
        s_prev: torch.Tensor = None
    ):
        """Returns (out, s_new).

        s_prev: carried associative state (B, H, D, D) for O(1) inference.
                h_prev kept for backward compat with CUDA recurrent scan path.
        """
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        cs = self.CHUNK_SIZE

        # 1. Projections
        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)
        else:
            q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        end_pos = start_pos + T
        if end_pos > self.rotary.max_seq_len_cached or self.rotary.cos_cached.device != q.device:
            new_len = max(end_pos, self.rotary.max_seq_len_cached * 2)
            self.rotary._build_cache(new_len, device=q.device)

        cos_tab = self.rotary.cos_cached[start_pos:end_pos].to(dtype=q.dtype)
        sin_tab = self.rotary.sin_cached[start_pos:end_pos].to(dtype=q.dtype)
        q_rot, k_rot = FusedRoPEELUFunction.apply(q.contiguous(), k.contiguous(), cos_tab, sin_tab)

        log_g = torch.log(torch.sigmoid(self.gamma_raw))

        # ------------------------------------------------------------------
        # O(1) INFERENCE PATH  (s_prev provided — single-token generation)
        # ------------------------------------------------------------------
        if s_prev is not None:
            gamma = torch.sigmoid(self.gamma_raw).to(dtype=x.dtype)
            state = s_prev
            outputs = []
            for t in range(T):
                q_t = q_rot[:, :, t, :]
                k_t = k_rot[:, :, t, :]
                v_t = v[:, :, t, :]
                kv    = torch.einsum('bhd,bhe->bhde', v_t, k_t)
                state = gamma.view(1, H, 1, 1) * state + kv
                z_t   = torch.einsum('bhde,bhe->bhd', state, q_t)
                outputs.append(z_t.unsqueeze(2))
            out_h = torch.cat(outputs, dim=2)
            out   = out_h.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out), state

        # ------------------------------------------------------------------
        # TRAINING / PREFILL — Path A: full-chunk batched
        # ------------------------------------------------------------------
        decay_mat   = torch.exp(log_g.view(H, 1, 1) * self._diff_clamp) * self._causal
        gamma_cross = torch.exp(log_g.view(H, 1) * self._i_idx_p1)
        gw          = torch.exp(log_g.view(H, 1) * self._c_m1_m_i)
        gamma_c     = torch.exp(log_g * cs)

        if T % cs == 0 and T >= cs:
            num_chunks = T // cs
            q_chunks = q_rot.view(B, H, num_chunks, cs, D).permute(0, 1, 2, 3, 4).reshape(B * H * num_chunks, cs, D)
            k_chunks = k_rot.view(B, H, num_chunks, cs, D).permute(0, 1, 2, 3, 4).reshape(B * H * num_chunks, cs, D)
            v_chunks = v.view(B, H, num_chunks, cs, D).permute(0, 1, 2, 3, 4).reshape(B * H * num_chunks, cs, D)

            raw      = torch.bmm(q_chunks, k_chunks.transpose(1, 2)).view(B, H, num_chunks, cs, cs)
            scores   = (raw * decay_mat.view(1, H, 1, cs, cs).to(dtype=raw.dtype)).view(B * H * num_chunks, cs, cs)
            intra_out = torch.bmm(scores, v_chunks).view(B, H, num_chunks, cs, D)

            v_w      = (v_chunks.view(B, H, num_chunks, cs, D) * gw.view(1, H, 1, cs, 1).to(dtype=v.dtype)).view(B * H * num_chunks, cs, D)
            delta_S  = torch.bmm(v_w.transpose(1, 2), k_chunks).view(B, H, num_chunks, D, D)

            all_states, h_last = RecurrentChunkStateScanFunction.apply(delta_S, gamma_c, h_prev)
            s_final = h_last  # final accumulated state after all chunks

            raw_cross  = torch.bmm(q_chunks, all_states.view(B * H * num_chunks, D, D).transpose(1, 2)).view(B, H, num_chunks, cs, D)
            cross_out  = raw_cross * gamma_cross.view(1, H, 1, cs, 1).to(dtype=raw_cross.dtype)

            total_out = (intra_out + cross_out).permute(0, 2, 3, 1, 4).reshape(B, T, C)
            return self.out_proj(total_out), s_final

        # Path B: arbitrary T
        state = h_prev.clone() if h_prev is not None else torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        for start in range(0, T, cs):
            end = min(start + cs, T)
            c   = end - start
            q_c = q_rot[:, :, start:end, :]
            k_c = k_rot[:, :, start:end, :]
            v_c = v[:, :, start:end, :]

            if c == cs:
                dm   = decay_mat
                gc_c = gamma_cross
                gw_c = gw
                gc_s = gamma_c.view(1, H, 1, 1)
            else:
                i_c      = self._i_idx[:c]
                diff_c   = (i_c.unsqueeze(1) - i_c.unsqueeze(0)).clamp(min=0)
                causal_c = (i_c.unsqueeze(1) - i_c.unsqueeze(0) >= 0).float()
                dm       = torch.exp(log_g.view(H, 1, 1) * diff_c) * causal_c
                gc_c     = torch.exp(log_g.view(H, 1) * (i_c + 1))
                gw_c     = torch.exp(log_g.view(H, 1) * (c - 1 - i_c))
                gc_s     = torch.exp(log_g * c).view(1, H, 1, 1)

            raw       = torch.einsum('bhid,bhjd->bhij', q_c, k_c)
            scores    = raw * dm.unsqueeze(0).to(dtype=raw.dtype)
            intra_out = torch.einsum('bhij,bhjd->bhid', scores, v_c)

            raw_cross = torch.einsum('bhde,bhie->bhid', state, q_c)
            cross_out = raw_cross * gc_c.unsqueeze(0).unsqueeze(-1).to(dtype=raw_cross.dtype)

            outputs.append(intra_out + cross_out)

            v_w       = v_c * gw_c.unsqueeze(0).unsqueeze(-1).to(dtype=v_c.dtype)
            chunk_upd = torch.einsum('bhid,bhie->bhde', v_w, k_c)
            state     = gc_s * state + chunk_upd

        out = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), state

