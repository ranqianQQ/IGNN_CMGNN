"""Fixed global class-compatibility correction for SFD-IGNN."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class GlobalCompatibilityCorrection(nn.Module):
    """Add one bounded graph-global compatibility residual to node logits."""

    def __init__(self, compatibility, initial_strength=.1, max_strength=4.):
        super().__init__()
        matrix = self._normalize(compatibility.detach().float())
        self.register_buffer("base_compatibility", matrix)
        self.matrix_residual = nn.Parameter(torch.zeros_like(matrix))
        self.max_strength = float(max_strength)
        ratio = min(max(initial_strength / max_strength, 1e-6), 1. - 1e-6)
        self.raw_strength = nn.Parameter(
            torch.tensor(math.log(ratio / (1. - ratio))))

    @staticmethod
    def _normalize(value):
        value = value.clamp_min(0.)
        return value / value.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _centered_log(probability):
        value = probability.clamp_min(1e-8).log()
        return value - value.mean(dim=1, keepdim=True)

    def compatibility(self):
        return (self.base_compatibility.clamp_min(1e-8).log()
                + self.matrix_residual).softmax(dim=1)

    def strength(self):
        return self.max_strength * torch.sigmoid(self.raw_strength)

    def regularization(self):
        return self.matrix_residual.square().mean()

    def forward(self, logits, adjacency, labels, train_mask):
        classes = logits.shape[1]
        probability = logits.softmax(dim=-1)
        seeded = probability.clone()
        seeded[train_mask] = F.one_hot(
            labels[train_mask], classes).to(probability.dtype)
        evidence = self._normalize(
            adjacency.matmul(seeded).matmul(self.compatibility().t()))
        return logits + self.strength() * self._centered_log(evidence)
