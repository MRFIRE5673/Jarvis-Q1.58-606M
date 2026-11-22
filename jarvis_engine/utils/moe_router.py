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
    Dot-product learned centroid router with Gumbel noise, STE top-K selection,
    and optional DeepSeek-V3 Auxiliary-Loss-Free load balancing via dynamic bias.
    """
    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2,
                 balance_alpha: float = 0.01, use_aux_free: bool = True,
                 bias_gamma: float = 0.005):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_alpha = balance_alpha
        self.use_aux_free = use_aux_free
        self.bias_gamma = bias_gamma
        self.expert_emb = nn.Parameter(
            torch.empty(num_experts, d_model).uniform_(-0.1, 0.1)
        )
        self.register_buffer("expert_bias", torch.zeros(num_experts))

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
        raw_scores = x @ self.expert_emb.t()  # [N, E]

        if self.use_aux_free:
            selection_scores = raw_scores + self.expert_bias
        else:
            selection_scores = raw_scores

        if self.training:
            selection_scores = selection_scores + self._gumbel(raw_scores.shape, raw_scores.device)

        probs = F.softmax(raw_scores, dim=-1)  # [N, E]

        # Top-K selection on selection_scores (with STE)
        mask = _STETopK.apply(selection_scores, self.top_k)  # [N, E] hard
        masked_scores = raw_scores.masked_fill(mask == 0, float("-inf"))
        topk_gates_full = F.softmax(masked_scores, dim=-1) * mask  # [N, E]
        topk_gates_full = topk_gates_full / (topk_gates_full.sum(-1, keepdim=True) + 1e-9)

        topk_gates, topk_idx = topk_gates_full.topk(self.top_k, dim=-1)  # [N, K]

        if self.use_aux_free:
            # DeepSeek-V3 dynamic closed-loop bias update (outside autograd)
            if self.training:
                with torch.no_grad():
                    f = mask.mean(dim=0)
                    target = 1.0 / self.num_experts
                    self.expert_bias.add_(-self.bias_gamma * (f - target))
            l_balance = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            # Traditional auxiliary load-balance loss: alpha * E * sum(f_i * P_i)
            f = mask.mean(dim=0)
            P = probs.mean(dim=0)
            l_balance = self.balance_alpha * self.num_experts * (f * P).sum()

        return topk_gates, topk_idx, l_balance


class TernaryMoE(nn.Module):
    """
    Complete Ternary MoE layer using SparseTopKRouter and TernaryLinear experts.
    Supports GELU and Squared ReLU (ReLU² = max(0, x)²).
    """
    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2,
                 hidden_mult: int = 2, balance_alpha: float = 0.01,
                 use_aux_free: bool = True, activation: str = "gelu"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation.lower()
        self.router = SparseTopKRouter(d_model, num_experts=num_experts, top_k=top_k,
                                       balance_alpha=balance_alpha, use_aux_free=use_aux_free)
        hidden = d_model * hidden_mult
        self.w1 = nn.ModuleList([TernaryLinear(d_model, hidden) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([TernaryLinear(hidden, d_model) for _ in range(num_experts)])

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation in ("relu2", "squared_relu"):
            return F.relu(x).square()
        return F.gelu(x)

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
                flat_out[m] = flat_gate[m].unsqueeze(-1) * self.w2[e](self._act(self.w1[e](xe)))

        out = flat_out.view(N, self.top_k, C).sum(dim=1).view(B, T, C)
        return out, l_balance
