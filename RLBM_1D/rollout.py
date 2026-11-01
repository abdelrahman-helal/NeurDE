"""
Pieces shared by Stage 2 training, eval.py and inspect_moments.py: the 16-node
starting states and 128-run targets (stage2.h5), one solver step with the network or
the classical equilibrium, and loading the frozen MomentMLP.

One network step:
  moments -> U, n, T (exact Landau frame of the state, as in NeurDE and the classical
  solver) -> network (+ MomentMLP estimates of the x/y features when the model takes
  them) -> f_eq = N0 feq_hat -> conservation projection -> tau -> Anderson-Witting
  collision -> streaming
"""

import h5py
import torch

import rlbm1d as R
from model import MomentMLP


class Stage2Data:
    def __init__(self, path, device, dtype):
        with h5py.File(path, "r") as h:
            self.f_init = torch.tensor(h["f16_init"][:], device=device, dtype=dtype)
            self.init_steps = h["init_steps"][:]
            self.fields = {k: torch.tensor(h[k][:], device=device, dtype=dtype) for k in ("n", "U0", "Uz", "T")}
            self.ratio = h.attrs["dt16"] / h.attrs["dt128"]  # 128-run steps per 16-node step
        self.n_steps = self.fields["n"].shape[0]

    def windows(self, rollout, lo, hi):
        """Starting states whose whole window lies inside 128-run steps [lo, hi]."""
        span = rollout * self.ratio
        return [k for k, i0 in enumerate(self.init_steps) if i0 >= lo and i0 + span <= hi]

    def target(self, name, t_index):
        return R.interp_time(self.fields[name], t_index)


def step_network(model, f, st, mlp=None):
    """One 16-node step with the network equilibrium. Returns the new state.
    mlp: the MomentMLP when the model takes the x/y moment features, else None."""
    mom = R.moments(f, st)
    U0, Uz, n, T = R.landau(*mom)
    x = R.features(*mom)
    if mlp is not None:
        x = torch.cat([x, mlp(x)], dim=1)
    feq_hat = model(x, st.basis)  # f_eq / N0
    feq = R.project_conservation(mom[0][None] * feq_hat.T, U0, Uz, *mom, st)
    return R.stream(R.collide(f, feq, U0, Uz, R.tau(n, T, st.v0), st), st)


def step_classical(f, st):
    """One 16-node step with the classical equilibrium: exact U from the moments and
    Alessandro's FEQ polynomial (the RLBM_STENCIL=16 run of relativistic_hydrodynamics)."""
    mom = R.moments(f, st)
    U0, Uz, n, T = R.landau(*mom)
    feq = R.poly_feq(st, n, U0, Uz, T)
    return R.stream(R.collide(f, feq, U0, Uz, R.tau(n, T, st.v0), st), st)


def load_moment_mlp(state, device, dtype):
    """Frozen MomentMLP from a train_moment_mlp.py checkpoint dict."""
    cm = state["cfg"]
    mlp = MomentMLP(cm["hidden"], cm["layers"], cm["activation"]).to(device, dtype)
    mlp.load_state_dict(state["model"])
    mlp.eval()
    return mlp.requires_grad_(False)
