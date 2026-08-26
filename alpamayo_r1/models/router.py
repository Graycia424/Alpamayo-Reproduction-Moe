"""Expert Router for MoE trajectory generation."""

import torch
import torch.nn.functional as F
from torch import nn


class ExpertRouter(nn.Module):
    """Routes inputs to experts based on CoT and image hidden states from VLM.

    Uses mean-pooled CoT and image hidden states, concatenated and passed through
    an MLP to produce gate logits. Training uses Straight-Through Gumbel-Softmax;
    inference uses argmax.
    """

    def __init__(self, vlm_hidden_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(vlm_hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self, cot_hidden: torch.Tensor, img_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing decision.

        Args:
            cot_hidden: [B, L_cot, D] CoT token hidden states from VLM last layer.
            img_hidden: [B, L_img, D] Image token hidden states from VLM last layer.

        Returns:
            gate_logits: [B, num_experts] raw logits for load-balancing loss.
            expert_idx: [B] selected expert index per sample.
        """
        cot_pool = cot_hidden.mean(dim=1)  # [B, D]
        img_pool = img_hidden.mean(dim=1)  # [B, D]
        gate_logits = self.gate(torch.cat([cot_pool, img_pool], dim=-1))  # [B, E]

        if self.training:
            # Straight-Through Gumbel-Softmax
            y_soft = F.gumbel_softmax(gate_logits, tau=1.0, hard=False)
            expert_idx = y_soft.argmax(dim=-1)
            # ST trick: hard forward, soft backward
            y_hard = F.one_hot(expert_idx, self.num_experts).float()
            self._gate_weights = y_hard - y_soft.detach() + y_soft
        else:
            expert_idx = gate_logits.argmax(dim=-1)

        return gate_logits, expert_idx
