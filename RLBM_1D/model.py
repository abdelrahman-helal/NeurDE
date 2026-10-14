"""
NeurDE for the 1D-reduced relativistic LBM.

Same structure as NeurDE/SOD_shock_tube/architectures/model.py:
    f_eq,i = N0 * exp( alpha(x) . phi(p_i) )
  alpha: DenseNet on the moments x = (N3, T00, T03, T33) / N0 (standardized)
  phi:   DenseNet on the node coordinates (p0/T0, mu)
The network itself returns feq_hat = f_eq / N0 (the density-scaled equilibrium; not
NeurDE's G population: the relativistic scheme has one distribution, so the network
learns its whole equilibrium). Multiplying by N0 makes f_eq scale with density exactly.
One addition: a linear head on alpha's features predicts vz = Uz/U0 (tanh keeps
|vz| < 1); U = gamma (1, vz).
"""

import torch
import torch.nn as nn

import rlbm1d  # noqa: F401  (puts NeurDE/SOD_shock_tube on sys.path)
from architectures.model import DenseNet  # noqa: E402


class NeurDE1D(nn.Module):
    def __init__(self, hidden=64, layers=4, activation="relu"):
        super().__init__()
        self.alpha = DenseNet([4] + [hidden] * layers, activation)
        self.phi = DenseNet([2] + [hidden] * layers, activation)
        self.vz_head = nn.Linear(hidden, 1)
        self.register_buffer("x_mean", torch.zeros(4))
        self.register_buffer("x_std", torch.ones(4))

    def set_normalization(self, x):
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp(min=1e-12))

    def forward(self, x, basis):
        """x: (S, 4) raw features; basis: (Q, 2). Returns feq_hat = f_eq / N0 (S, Q) and vz (S,)."""
        a = self.alpha((x - self.x_mean) / self.x_std)
        feq_hat = torch.exp(a @ self.phi(basis).T)
        vz = torch.tanh(self.vz_head(a)).squeeze(-1)
        return feq_hat, vz
