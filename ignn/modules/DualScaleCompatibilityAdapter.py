"""Global compatibility propagation with a gated node-wise residual."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class DualScaleCompatibilityAdapter(nn.Module):
    """Fuse conservative class compatibility and node-specific corrections.

    The global branch is the low-variance compatibility propagation path.  A
    shared relation MLP models only the residual left by that branch.  Both
    branches use the same compatibility matrix, and a learned bounded gate
    controls how much node-wise capacity is admitted.
    """

    def __init__(self, compatibility, hidden=8, dropout=.1,
                 initial_strength=.1, max_strength=4., initial_gate=.1):
        super().__init__()
        compatibility = self._normalize(compatibility.detach().float())
        self.register_buffer("base_compatibility", compatibility.clone())
        self.matrix_residual = nn.Parameter(torch.zeros_like(compatibility))
        self.max_strength = float(max_strength)
        strength_ratio = min(max(initial_strength / max_strength, 1e-6), 1-1e-6)
        self.raw_strength = nn.Parameter(torch.tensor(
            math.log(strength_ratio / (1-strength_ratio))))
        gate_ratio = min(max(float(initial_gate), 1e-6), 1-1e-6)
        self.raw_gate = nn.Parameter(torch.tensor(
            math.log(gate_ratio / (1-gate_ratio))))
        classes = compatibility.shape[0]
        self.relation = nn.Sequential(
            nn.Linear(4 * classes, int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), classes),
        )
        nn.init.zeros_(self.relation[-1].weight)
        nn.init.zeros_(self.relation[-1].bias)

    @staticmethod
    def _normalize(value):
        value = value.clamp_min(0.)
        return value / value.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def compatibility(self):
        return (self.base_compatibility.clamp_min(1e-8).log()
                + self.matrix_residual).softmax(dim=1)

    def strength(self):
        return self.max_strength * self.raw_strength.sigmoid()

    def gate(self):
        return self.raw_gate.sigmoid()

    def regularization(self):
        return self.matrix_residual.square().mean()

    def components(self, logits, adjacency, labels, train_mask):
        classes = logits.shape[1]
        probability = logits.softmax(dim=-1)
        seeded = probability.clone()
        seeded[train_mask] = F.one_hot(
            labels[train_mask], classes).to(probability.dtype)
        evidence = adjacency.matmul(seeded).matmul(self.compatibility().t())
        evidence = self._normalize(evidence)
        base_log = probability.clamp_min(1e-8).log()
        base_log = base_log - base_log.mean(dim=1, keepdim=True)
        evidence_log = evidence.clamp_min(1e-8).log()
        evidence_log = evidence_log - evidence_log.mean(dim=1, keepdim=True)
        global_logits = logits + self.strength() * evidence_log
        features = torch.cat([
            base_log,
            evidence_log,
            (base_log - evidence_log).abs(),
            base_log * evidence_log,
        ], dim=1)
        residual = self.relation(features)
        corrected = global_logits + self.gate() * residual
        return corrected, global_logits, residual

    def forward(self, logits, adjacency, labels, train_mask):
        return self.components(logits, adjacency, labels, train_mask)[0]
