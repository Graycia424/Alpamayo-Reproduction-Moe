"""Load balancing loss for MoE routing."""

import torch
import torch.nn.functional as F


def load_balancing_loss(gate_logits: torch.Tensor, expert_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch Transformer style load balancing loss.

    Encourages uniform expert selection across the batch.

    Args:
        gate_logits: [B, num_experts] raw router logits.
        expert_idx: [B] selected expert indices.
        num_experts: total number of experts.

    Returns:
        Scalar loss tensor.
    """
    # f_i: fraction of samples routed to expert i
    f = torch.zeros(num_experts, device=gate_logits.device)
    f.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=f.dtype))
    f = f / expert_idx.shape[0]

    # p_i: mean router probability for expert i
    p = F.softmax(gate_logits, dim=-1).mean(dim=0)

    return num_experts * (f * p).sum()
