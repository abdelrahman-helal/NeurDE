"""
NeurDE for the 1D-reduced relativistic LBM.

Same structure as NeurDE/SOD_shock_tube/architectures/model.py:
    f_eq,i = N0 * exp( alpha(x) . phi(p_i) )
  alpha: DenseNet on the moments x (standardized): the 4 features (N3, T00, T03, T33) / N0,
         optionally followed by the 8 x/y features of rlbm1d.xy_features (n_in = 12)
  phi:   DenseNet on the node coordinates (p0/T0, mu)
The network itself returns feq_hat = f_eq / N0 (the density-scaled equilibrium; not
NeurDE's G population: the relativistic scheme has one distribution, so the network
learns its whole equilibrium). Multiplying by N0 makes f_eq scale with density exactly.
As in NeurDE, the velocity U is not predicted: the solver takes it from the state's moments.

MomentMLP estimates the 8 x/y features from the 4 t/z features. A reduced set has no
x/y entries, so in a rollout these estimates stand in for the 128 run's values.
"""

import torch
import torch.nn as nn

import rlbm1d  # noqa: F401  (puts NeurDE/SOD_shock_tube on sys.path)
from architectures.model import DenseNet  # noqa: E402


class NeurDE1D(nn.Module):
    def __init__(self, hidden=64, layers=4, activation="relu", n_in=4):
        super().__init__()
        self.alpha = DenseNet([n_in] + [hidden] * layers, activation)
        self.phi = DenseNet([2] + [hidden] * layers, activation)
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))

    def set_normalization(self, x):
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp(min=1e-12))

    def forward(self, x, basis):
        """x: (S, n_in) raw features; basis: (Q, 2). Returns feq_hat = f_eq / N0 (S, Q)."""
        a = self.alpha((x - self.x_mean) / self.x_std)
        return torch.exp(a @ self.phi(basis).T)


class MomentMLP(nn.Module):
    """The 8 x/y features from the 4 t/z features; inputs standardized, outputs learned
    in units of each target's standard deviation (the targets are ~1e-3)."""

    def __init__(self, hidden=64, layers=4, activation="relu"):
        super().__init__()
        self.net = DenseNet([4] + [hidden] * layers + [8], activation)
        self.register_buffer("x_mean", torch.zeros(4))
        self.register_buffer("x_std", torch.ones(4))
        self.register_buffer("y_std", torch.ones(8))

    def set_normalization(self, x, y):
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp(min=1e-12))
        self.y_std.copy_(y.std(0).clamp(min=1e-30))

    def forward(self, x):
        """x: (S, 4) raw t/z features. Returns the 8 x/y features (S, 8)."""
        return self.net((x - self.x_mean) / self.x_std) * self.y_std
