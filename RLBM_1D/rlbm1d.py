"""
Shared pieces for NeurDE on the 1D-reduced relativistic LBM (TRAINING_PLAN.md).

  * Velocity set: reduced_stencil.ReducedStencil from relativistic_hydrodynamics
    (one source of truth for the 16/24-node sets). Default: hops (2, 5), T0 = 0.75.
  * Moments: 1D monomials p0^(k-b) pz^b; f carries the quadrature weight, so
    N^a = sum_i f_i p_i^a as in relativistic_hydrodynamics/collide.py.
  * EntropicFit: positive f_i = w_i exp(lambda . phi_i) matching given moments
    (Stage 1 targets, Stage 2 starting states).
  * 1D solver in torch, arrays of shape (Q, S): moments, exact Landau-frame U/n/T
    (closed-form 2x2 eigenproblem), relaxation time, conservation projection,
    Anderson-Witting collision, streaming. Every formula follows collide.py and
    propagate.py; test_solver1d.py checks this against Alessandro's code.
"""

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RLBM_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "relativistic_hydrodynamics"))
SOD_DIR = os.path.normpath(os.path.join(HERE, "..", "SOD_shock_tube"))
# Appended, not prepended: files in this folder (train_stage_2.py, eval.py, ...) must
# win over SOD_shock_tube's files of the same name
for _p in (RLBM_DIR, SOD_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from reduced_stencil import ReducedStencil  # noqa: E402  (does not import config)

# Grid and relaxation-time constants of the 128-velocity run (config.py, collide.py)
DX = 12.8 / 12800
V0_128 = 1.0 / math.sqrt(41.0)
DT_128 = DX / V0_128
ETA_S = 0.1
ETA_KCE = 4.0 / 5.0
HBAR = 197.3269788


def monomial_list(kmax):
    """(k, b) for monomials p0^(k-b) pz^b of degree k <= kmax, ordered by degree."""
    return [(k, b) for k in range(kmax + 1) for b in range(k + 1)]


def monomials(p0, pz, kb):
    """Rows = monomials, columns = nodes. Works for numpy arrays and torch tensors."""
    rows = [p0 ** (k - b) * pz ** b for k, b in kb]
    return torch.stack(rows) if isinstance(p0, torch.Tensor) else np.stack(rows)


class Stencil1D:
    """The reduced velocity set as numpy arrays and as torch tensors on `device`."""

    def __init__(self, hops=(2, 5), T0=0.75, device="cpu", dtype=torch.float64):
        st = ReducedStencil(tuple(hops), T0=T0, device="cpu", dtype=torch.float64)
        self.hops, self.T0, self.v0, self.Q, self.name = tuple(hops), T0, st.v0, st.NPOP, st.name
        self.dt = DX / self.v0
        self.np = {k: getattr(st, k).numpy() for k in ("p0", "pz", "p_perp", "mu", "w")}
        self.nz = [int(n) for n in st.nz]
        t = lambda a: torch.tensor(a, dtype=dtype, device=device)
        self.p0, self.pz, self.p_perp, self.mu, self.w = (t(self.np[k]) for k in ("p0", "pz", "p_perp", "mu", "w"))
        # Node coordinates fed to NeurDE's trunk network phi
        self.basis = torch.stack([self.p0 / T0, self.mu], dim=1)


# ------------------------------------------------------------------ entropic fit

class EntropicFit:
    """Positive f_i = w_i exp(lambda . phi_i) whose monomial moments equal targets.

    phi is a w-orthonormal basis of the given monomials (better conditioning; the
    family of distributions is the same). Damped Newton on the convex dual
    L(lambda) = sum_i f_i - lambda . targets, whose gradient is (moments - targets).
    """

    def __init__(self, p0, pz, w, kb):
        self.w = w
        self.A = monomials(p0, pz, kb)
        D = 1.0 / np.linalg.norm(self.A, axis=1)
        R = np.linalg.qr((self.A * D[:, None] * np.sqrt(w)).T, mode="r")
        R = R * np.sign(np.diag(R))[:, None]
        self.C = np.linalg.solve(R.T, np.diag(D))  # phi = C @ A
        self.phi = self.C @ self.A
        self.r = self.A.shape[0]

    def f(self, lam):
        return self.w * np.exp(np.clip(lam @ self.phi, -700, 700))

    def solve(self, M, lead, lam0=None, maxit=100, tol=1e-10, chunk=200_000):
        """M: target moments (r, S); lead: scale per row and site (r, S) for the
        convergence test. Returns f (Q, S), lambda (S, r), converged (S,)."""
        S = M.shape[1]
        lam = np.zeros((S, self.r)) if lam0 is None else lam0.copy()
        if lam0 is None:
            lam[:, 0] = np.log(np.maximum((self.C @ M)[0], 1e-300))
        for c0 in range(0, S, chunk):
            sl = slice(c0, min(S, c0 + chunk))
            lam[sl] = self._solve_chunk(M[:, sl], lead[:, sl], lam[sl], maxit, tol)
        f = self.f(lam)
        conv = (np.abs(f @ self.A.T - M.T) / lead.T).max(1) < 1e-8
        return f.T, lam, conv

    def _solve_chunk(self, M, lead, lam, maxit, tol):
        Tg = (self.C @ M).T
        active = np.ones(lam.shape[0], bool)
        for _ in range(maxit):
            ia = np.flatnonzero(active)
            if ia.size == 0:
                break
            fa = self.f(lam[ia])
            res = np.abs(fa @ self.A.T - M[:, ia].T) / lead[:, ia].T
            done = res.max(1) < tol
            active[ia[done]] = False
            ia, fa = ia[~done], fa[~done]
            if ia.size == 0:
                break
            g = fa @ self.phi.T - Tg[ia]
            H = np.einsum("si,ai,bi->sab", fa, self.phi, self.phi)
            H += 1e-14 * np.einsum("saa->s", H)[:, None, None] * np.eye(self.r)
            step = np.linalg.solve(H, g[..., None])[..., 0]
            L0 = fa.sum(1) - (lam[ia] * Tg[ia]).sum(1)
            slope = (g * step).sum(1)
            accepted = np.zeros(ia.size, bool)
            for alpha in 0.5 ** np.arange(0, 30):
                todo = ~accepted
                if not todo.any():
                    break
                trial = lam[ia[todo]] - alpha * step[todo]
                L1 = self.f(trial).sum(1) - (trial * Tg[ia[todo]]).sum(1)
                ok = L1 <= L0[todo] - 1e-4 * alpha * slope[todo] + 1e-15 * np.abs(L0[todo])
                sub = np.flatnonzero(todo)[ok]
                lam[ia[sub]] = trial[ok]
                accepted[sub] = True
        return lam


# ------------------------------------------------------------------- 1D solver

def moments(f, st):
    """N0, N3, T00, T03, T33 of f (Q, S); px = py = 0 on a reduced set."""
    p0, pz = st.p0[:, None], st.pz[:, None]
    return (f * p0).sum(0), (f * pz).sum(0), (f * p0 * p0).sum(0), (f * p0 * pz).sum(0), (f * pz * pz).sum(0)


def features(N0, N3, T00, T03, T33):
    """Network input: the 1D moments divided by N0 (f_eq scales linearly with N0)."""
    return torch.stack([N3 / N0, T00 / N0, T03 / N0, T33 / N0], dim=1)


def landau(N0, N3, T00, T03, T33):
    """Exact Landau-frame U, n, T: largest eigenvalue of [[T00, -T03], [T03, -T33]]
    (the mixed tensor of collide.py with Ux = Uy = 0), eigenvector, n = U.N, T = eps/(3n)."""
    eps = 0.5 * ((T00 - T33) + torch.sqrt(torch.clamp((T00 + T33) ** 2 - 4 * T03 ** 2, min=0)))
    vz = T03 / (eps + T33)  # second eigenvector row; stable as T03 -> 0
    U0 = 1 / torch.sqrt(1 - vz ** 2)
    Uz = U0 * vz
    n = torch.clamp(U0 * N0 - Uz * N3, min=1e-30)
    return U0, Uz, n, eps / (3 * n)


def n_eps_T(U0, Uz, N0, N3, T00, T03, T33):
    """n = U.N, eps = U_a T^ab U_b, T = eps/(3n) for a given (e.g. predicted) U."""
    n = torch.clamp(U0 * N0 - Uz * N3, min=1e-30)
    eps = U0 * U0 * T00 - 2 * U0 * Uz * T03 + Uz * Uz * T33
    return n, eps, eps / (3 * n)


def tau(n, T, v0):
    """Local relaxation time, as in collide.py (BAMPS gluon entropy, eta/s = ETA_S)."""
    rho_eq = 16.0 * T ** 3 / math.pi ** 2
    s = 4.0 * n - n * torch.log(n / rho_eq)
    eta = ETA_KCE * n * T
    return 0.5 + ETA_S * s * (HBAR / 400.0) * (v0 / DX) / eta


def project_conservation(feq, U0, Uz, N0, N3, T00, T03, T33, st):
    """Smallest relative change of feq such that the AW collision conserves N0, T00, T0z:
        sum (p.U) feq = U_b N^b,   sum (p.U) p^a feq = T^ab U_b  (a = 0, z).
    feq <- feq * (1 + C^T lambda), so zero stays zero and small changes keep feq > 0."""
    pU = st.p0[:, None] * U0 - st.pz[:, None] * Uz                     # (Q, S)
    C = torch.stack([pU, pU * st.p0[:, None], pU * st.pz[:, None]])     # (3, Q, S)
    b = torch.stack([U0 * N0 - Uz * N3, U0 * T00 - Uz * T03, U0 * T03 - Uz * T33])  # (3, S)
    r = b - torch.einsum("aqs,qs->as", C, feq)
    G = torch.einsum("aqs,qs,bqs->sab", C, feq, C)
    lam = torch.linalg.solve(G, r.T.unsqueeze(-1)).squeeze(-1)          # (S, 3)
    return feq * (1 + torch.einsum("aqs,sa->qs", C, lam))


def collide(f, feq, U0, Uz, tau_, st):
    """Anderson-Witting relaxation, as in collide.py: UL = U.p / p0."""
    UL = U0 - Uz * st.pz[:, None] / st.p0[:, None]
    return f - (UL / tau_) * (f - feq)


def stream(f, st):
    """Periodic streaming along z by each population's hop (propagate.py)."""
    return torch.stack([torch.roll(f[i], st.nz[i], dims=-1) for i in range(st.Q)])


def poly_feq(st, n, U0, Uz, T):
    """Classical equilibrium: Alessandro's FEQ polynomial in T0 units at the reduced
    nodes (same as quadrature.compute_feq on a reduced stencil)."""
    from quadrature import compute_feq_expression
    T0 = st.T0
    col = lambda a: (a / T0)[:, None]
    expr = compute_feq_expression(col(st.p0), col(st.p_perp), col(torch.zeros_like(st.p0)), col(st.pz),
                                  U0[None], torch.zeros_like(U0)[None], torch.zeros_like(U0)[None],
                                  Uz[None], (T / T0)[None])
    return st.w[:, None] * n[None] * expr / T0


def relL2(a, b):
    return torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp(min=1e-300)


def interp_time(fields, t_index):
    """Linear interpolation of a (T, ...) array of 128-run fields at fractional index."""
    i = min(int(math.floor(t_index)), fields.shape[0] - 2)
    a = t_index - i
    return (1 - a) * fields[i] + a * fields[i + 1]
