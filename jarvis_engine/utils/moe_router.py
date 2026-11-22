"""
Sparse Top-K MoE Router with Straight-Through Estimator (STE) and Gumbel Noise.
Adapted from author's friend's implementation for Jarvis-Q1.58.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from jarvis_engine.utils.ternary_ops import TernaryLinear


class _STETopK(torch.autograd.Function):
    """
    Straight-Through Top-K Estimator.
    Forward: hard binary top-K mask.
    Backward: straight-through pass of gradients only to selected positions.
    """
    @staticmethod
    def forward(ctx, scores: torch.Tensor, K: int) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(scores)
        ctx.K = K
        _, idx = torch.topk(scores, K, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask.float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (scores,) = ctx.saved_tensors
        _, idx = torch.topk(scores, ctx.K, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return grad_output * mask.float(), None


class SparseTopKRouter(nn.Module):
    """
    Dot-product learned centroid router with Gumbel noise and STE top-K selection.
    """
    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2,
                 balance_alpha: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_alpha = balance_alpha
        self.expert_emb = nn.Parameter(
            torch.empty(num_experts, d_model).uniform_(-0.1, 0.1)
        )

    @staticmethod
    def _gumbel(shape: torch.Size, device: torch.device) -> torch.Tensor:
        u = torch.rand(shape, device=device).clamp(min=1e-10)
        return -torch.log(-torch.log(u))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, d_model] flattened token representations
        Returns:
            topk_gates: [N, top_k] normalized dispatch gate weights
            topk_idx:   [N, top_k] expert indices
            l_balance:  scalar load balancing loss
        """
        N = x.size(0)
        scores = x @ self.expert_emb.t()  # [N, E]

        if self.training:
            scores = scores + self._gumbel(scores.shape, scores.device)

        probs = F.softmax(scores, dim=-1)  # [N, E]

        mask = _STETopK.apply(scores, self.top_k)  # [N, E] hard
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        topk_gates_full = F.softmax(masked_scores, dim=-1) * mask  # [N, E]
        topk_gates_full = topk_gates_full / (topk_gates_full.sum(-1, keepdim=True) + 1e-9)

        topk_gates, topk_idx = topk_gates_full.topk(self.top_k, dim=-1)  # [N, K]

        # Load-balance loss: alpha * E * sum(f_i * P_i) (Paper Eq. 6)
        f = mask.mean(dim=0)
        P = probs.mean(dim=0)
        l_balance = self.balance_alpha * self.num_experts * (f * P).sum()

        return topk_gates, topk_idx, l_balance


class TernaryMoE(nn.Module):
    """
    Complete Ternary MoE layer using SparseTopKRouter and TernaryLinear experts.
    """
    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2,
                 hidden_mult: int = 2, balance_alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = SparseTopKRouter(d_model, num_experts=num_experts, top_k=top_k,
                                       balance_alpha=balance_alpha)
        hidden = d_model * hidden_mult
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        topk_gates, topk_idx, l_balance = self.router(x_flat)

        flat_idx = topk_idx.reshape(-1)
        flat_gate = topk_gates.reshape(-1)
        flat_x = x_flat.repeat_interleave(self.top_k, dim=0)
        flat_out = torch.zeros_like(flat_x)

        for e in range(self.num_experts):
            m = flat_idx == e
            if m.any():
                xe = flat_x[m]
                flat_out[m] = flat_gate[m].unsqueeze(-1) * self.w2[e](F.gelu(self.w1[e](xe)))

        out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)
        return out, l_balance
