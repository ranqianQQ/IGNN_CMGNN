"""Parameter-efficient spectral feature decoupling for c-IGNN."""
import torch
from torch import nn


class SpectralFeatureDecoupling(nn.Module):
    """Mix signed consecutive-hop detail into each hop before concat NR."""

    def __init__(self, base, n_hops, width):
        super().__init__()
        self.base = base
        self.n_tokens = n_hops + 1
        self.width = width
        self.detail_gates = nn.Parameter(torch.zeros(n_hops, width))

    def reset_decoupling_parameters(self):
        nn.init.zeros_(self.detail_gates)

    def coefficients(self):
        return self.detail_gates.tanh()

    def forward(self, x):
        hops = x.reshape(x.shape[0], self.n_tokens, self.width)
        if self.n_tokens == 1:
            return self.base(x)
        details = hops[:, 1:] - hops[:, :-1]
        refined = torch.cat(
            (hops[:, :1], hops[:, 1:] + self.coefficients().unsqueeze(0) * details), dim=1
        )
        return self.base(refined.flatten(1))
