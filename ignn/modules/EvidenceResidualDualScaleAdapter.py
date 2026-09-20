"""Interpretable global and node-local compatibility corrections."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class EvidenceResidualDualScaleAdapter(nn.Module):
    """Correct logits at class-global and node-local scales.

    The global term uses the same compatibility evidence for every node.  The
    local term is deliberately constrained: it can only increase or decrease
    the strength of that same compatibility evidence for each node.  Its
    signed coefficients are centred across nodes, so the global term controls
    the population mean while the local term models zero-mean deviations.
    This prevents the local branch from becoming a second classifier and gives
    the two scales identifiable roles.
    """

    MODES = ("global", "local", "dual")

    def __init__(self, compatibility, hidden=8, dropout=.1,
                 initial_global_strength=.1, max_global_strength=4.,
                 max_local_strength=1., classwise_gate=False):
        super().__init__()
        compatibility = self._normalize(compatibility.detach().float())
        self.register_buffer("base_compatibility", compatibility.clone())
        self.matrix_residual = nn.Parameter(torch.zeros_like(compatibility))
        self.max_global_strength = float(max_global_strength)
        self.max_local_strength = float(max_local_strength)
        ratio = min(max(initial_global_strength / max_global_strength, 1e-6),
                    1. - 1e-6)
        self.raw_global_strength = nn.Parameter(torch.tensor(
            math.log(ratio / (1. - ratio))))
        classes = compatibility.shape[0]
        gate_width = classes if classwise_gate else 1
        self.classwise_gate = bool(classwise_gate)
        self.local_scorer = nn.Sequential(
            nn.Linear(4 * classes, int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), gate_width),
        )
        # Exact global-only initialization for dual mode, and exact backbone
        # initialization for local-only mode.
        nn.init.zeros_(self.local_scorer[-1].weight)
        nn.init.zeros_(self.local_scorer[-1].bias)

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

    def global_strength(self):
        return self.max_global_strength * self.raw_global_strength.sigmoid()

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

        base_log = self._centered_log(probability)
        evidence_log = self._centered_log(evidence)
        disagreement = evidence_log - base_log
        relation = torch.cat([
            base_log,
            evidence_log,
            disagreement.abs(),
            base_log * evidence_log,
        ], dim=1)
        raw_local_coefficient = torch.tanh(self.local_scorer(relation))
        # Dividing by two preserves the configured [-max, max] bound after
        # subtracting the node mean (two values in [-1, 1] can differ by two).
        local_coefficient = (.5 * self.max_local_strength
                             * (raw_local_coefficient
                                - raw_local_coefficient.mean(dim=0,
                                                             keepdim=True)))
        local_residual = local_coefficient * evidence_log
        global_residual = self.global_strength() * evidence_log
        return {
            "global": logits + global_residual,
            "local": logits + local_residual,
            "dual": logits + global_residual + local_residual,
            "global_residual": global_residual,
            "local_residual": local_residual,
            "local_coefficient": local_coefficient,
            "evidence": evidence,
        }

    def forward(self, logits, adjacency, labels, train_mask, mode="global"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown correction mode: {mode}")
        return self.components(logits, adjacency, labels, train_mask)[mode]
