"""Compatibility-guided classifier fine-tuning on top of an IGNN backbone."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class CompatibilityGuidedFineTuning(nn.Module):
    """Turn a CMGNN compatibility estimate into a trainable logit adapter.

    ``M[a, b]`` describes how likely a class-b neighbour is for a class-a
    centre. Neighbour beliefs are mapped back to centre-class evidence with
    ``M.T`` and fused with the original classifier in log space.
    """

    def __init__(self, compatibility, initial_strength=0.1, max_strength=4.0,
                 learnable_matrix=True):
        super().__init__()
        compatibility = self._normalize(compatibility.detach().float())
        self.register_buffer('base_compatibility', compatibility.clone())
        self.matrix_residual = nn.Parameter(
            torch.zeros_like(compatibility), requires_grad=learnable_matrix)
        self.max_strength = float(max_strength)
        ratio = min(max(float(initial_strength) / self.max_strength, 1e-6), 1 - 1e-6)
        self.raw_strength = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    @staticmethod
    def _normalize(matrix):
        return matrix.clamp_min(0.0) / matrix.clamp_min(0.0).sum(
            dim=1, keepdim=True).clamp_min(1e-12)

    def strength(self):
        return self.max_strength * self.raw_strength.sigmoid()

    def compatibility(self):
        prior_logits = self.base_compatibility.clamp_min(1e-8).log()
        return (prior_logits + self.matrix_residual).softmax(dim=1)

    @torch.no_grad()
    def update_compatibility(self, estimate, momentum=0.8):
        estimate = self._normalize(estimate.to(self.base_compatibility))
        updated = float(momentum) * self.base_compatibility + (1.0 - float(momentum)) * estimate
        self.base_compatibility.copy_(self._normalize(updated))

    def regularization(self):
        return self.matrix_residual.square().mean()

    def forward(self, logits, adjacency, labels, train_mask, steps=1):
        base_logits = logits
        beliefs = logits.softmax(dim=-1)
        train_targets = F.one_hot(
            labels[train_mask], num_classes=logits.shape[1]).to(beliefs.dtype)
        compatibility = self.compatibility()
        corrected = base_logits
        for _ in range(int(steps)):
            seeded = beliefs.clone()
            seeded[train_mask] = train_targets
            neighbour_beliefs = adjacency.matmul(seeded)
            centre_evidence = neighbour_beliefs.matmul(compatibility.t())
            centre_evidence = centre_evidence / centre_evidence.sum(
                dim=1, keepdim=True).clamp_min(1e-12)
            log_evidence = centre_evidence.clamp_min(1e-8).log()
            log_evidence = log_evidence - log_evidence.mean(dim=1, keepdim=True)
            corrected = base_logits + self.strength() * log_evidence
            beliefs = corrected.softmax(dim=-1)
        return corrected

