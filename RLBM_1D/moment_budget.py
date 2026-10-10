"""
Phase 0: offline higher-moment budget (nothing is trained or simulated).

Every moment is a fixed linear sum over the nodes, M = sum_i g_i p0_i^(k-b) pz_i^b,
so with only Alessandro's 128-velocity h5 and the candidate 16-node stencils we can
measure in advance:

  0a. which moments the 128 run actually carries (equilibrium feq and the
      non-equilibrium part f - feq), degree k = 0..5, as size relative to the
      leading moment of the same degree;
  0b. which energy shells carry each moment versus which shells dominate a plain
      relative-L2 loss on f (the imbalance NeurDE's loss would have);
  0c. the ceiling: for each candidate 16-node stencil, the best positive 16-node f
      (entropic form, f_i = w_i exp(lambda . phi_i)) that matches the 128 feq moments
      up to degree K = 1..4, and its error on the moments it does not match. Also
      the 3-constraint Landau target of the earlier plan and Alessandro's FEQ
      polynomial evaluated on the 16 nodes;
  0d. a printed recommendation (target degree K, stencil).

All reference values come from the 128-velocity data, never from analytic
Maxwell-Juettner integrals.

Moments use the code's convention: f already carries the quadrature weight, so
N^a = sum_i f_i p_i^a (see relativistic_hydrodynamics/collide.py).

Usage (from this folder):
    python moment_budget.py
    python moment_budget.py --quick            # few steps/sites, for a smoke test
Writes results/phase0_report.md and results/phase0_summary.json.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RLBM_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "relativistic_hydrodynamics"))
sys.path.insert(0, HERE)
sys.path.insert(0, RLBM_DIR)

from stencil16 import Stencil16  # noqa: E402

KMAX = 5  # highest moment degree reported
KB = [(k, b) for k in range(KMAX + 1) for b in range(k + 1)]  # monomial p0^(k-b) pz^b
DEG = np.array([k for k, _ in KB])
NONEQ_THRESHOLD = 1e-6  # sum|f - feq| / sum|feq| above this -> "non-eq" site


# ----------------------------------------------------------------------------- data

def load_stencil128():
    from stencil import Stencil  # relativistic_hydrodynamics/stencil.py
    s = Stencil()
    return {k: getattr(s, k).cpu().numpy() for k in ("p0", "px", "py", "pz", "w")}


def monomials(p0, pz, kb=KB):
    """Rows = monomials p0^(k-b) pz^b, columns = nodes."""
    return np.stack([p0 ** (k - b) * pz ** b for k, b in kb])


def load_samples(h5path, n_steps, max_eq, max_noneq, seed):
    """Sample sites from evenly spaced steps: all kinds of states, both site groups."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("feq", "f", "rho", "T", "U0", "Ux", "Uy", "Uz", "noneq", "step")}
    with h5py.File(h5path, "r") as h:
        n_t = h["feq"].shape[0]
        steps = np.unique(np.linspace(0, n_t - 1, n_steps).astype(int))
        iters = h["iterations"][:]
        for t in steps:
            feq = h["feq"][t].reshape(128, -1)
            f = h["f"][t].reshape(128, -1)
            r = np.abs(f - feq).sum(0) / np.abs(feq).sum(0)
            noneq_idx = np.flatnonzero(r > NONEQ_THRESHOLD)
            eq_idx = np.flatnonzero(r <= NONEQ_THRESHOLD)
            if noneq_idx.size > max_noneq:
                noneq_idx = rng.choice(noneq_idx, max_noneq, replace=False)
            if eq_idx.size > max_eq:
                eq_idx = rng.choice(eq_idx, max_eq, replace=False)
            idx = np.sort(np.concatenate([noneq_idx, eq_idx]))
            out["feq"].append(feq[:, idx])
            out["f"].append(f[:, idx])
            for key in ("rho", "T", "U0", "Ux", "Uy", "Uz"):
                out[key].append(h[key][t].reshape(-1)[idx])
            out["noneq"].append(r[idx] > NONEQ_THRESHOLD)
            out["step"].append(np.full(idx.size, iters[t]))
    data = {k: np.concatenate(v, axis=-1) for k, v in out.items()}
    data["steps"] = iters[steps]
    return data


# --------------------------------------------------------------------- statistics

def degree_error(M_pred, M_ref, lead):
    """|M_pred - M_ref| / leading moment of the same degree, worst monomial per degree.
    Returns (KMAX+1, S)."""
    rel = np.abs(M_pred - M_ref) / lead[DEG]
    return np.stack([rel[DEG == k].max(0) for k in range(KMAX + 1)])


def stats(x, mask):
    x = x[..., mask]
    if x.shape[-1] == 0:
        return dict(median=np.nan, p95=np.nan, max=np.nan)
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))


def fmt(v):
    return "  -   " if v is None or not np.isfinite(v) else f"{v:.1e}"


# ------------------------------------------------------------ entropic moment fit

class EntropicFit:
    """f_i = w_i exp(lambda . phi_i), phi = w-orthonormal basis of the monomials of
    degree <= K that are linearly independent on the stencil nodes. Bases are nested
    in K (lower-triangular change of basis), so lambda_K warm-starts lambda_{K+1}."""

    def __init__(self, st, K):
        self.st = st
        A_all = monomials(st.p0, st.pz)
        sw = np.sqrt(st.w)
        keep, rows = [], []
        for m, (k, b) in enumerate(KB):
            if k > K:
                break
            cand = rows + [A_all[m] / np.linalg.norm(A_all[m])]
            R = np.linalg.qr((np.stack(cand) * sw).T, mode="r")
            if abs(R[-1, -1]) > 1e-9:
                keep.append(m)
                rows = cand
        self.sel = np.array(keep)
        D = np.array([1 / np.linalg.norm(A_all[m]) for m in keep])
        A_n = A_all[self.sel] * D[:, None]
        R = np.linalg.qr((A_n * sw).T, mode="r")
        sign = np.sign(np.diag(R))
        R = R * sign[:, None]
        self.C = np.linalg.solve(R.T, np.diag(D))  # phi = C @ A_sel
        self.phi = self.C @ A_all[self.sel]
        self.r = len(keep)

    def f(self, lam):
        return self.st.w * np.exp(np.clip(lam @ self.phi, -700, 700))

    def solve(self, M_sel, lead_sel, lam0, maxit=200, tol=1e-10):
        """Damped Newton on the convex dual  L = sum_i f_i - lam . T."""
        Tg = (self.C @ M_sel).T  # (S, r)
        lam = lam0.copy()
        S = lam.shape[0]
        active = np.ones(S, bool)
        A_sel = self.C_inv_rows()
        for _ in range(maxit):
            ia = np.flatnonzero(active)
            if ia.size == 0:
                break
            fa = self.f(lam[ia])
            res = np.abs(fa @ A_sel.T - M_sel[:, ia].T) / lead_sel[ia]
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
        fa = self.f(lam)
        res = np.abs(fa @ A_sel.T - M_sel.T) / lead_sel
        return lam, res.max(1) < 1e-8

    def C_inv_rows(self):
        return monomials(self.st.p0, self.st.pz)[self.sel]


def landau_fit(st, fit1, lam1, U0, Uz, M128_landau):
    """Earlier plan's target: exp(lambda . phi_K=1) matching the 3 Landau/AW-conservation
    constraints  sum (p.U) f,  sum (p.U) p0 f,  sum (p.U) pz f  of the 128 feq."""
    pU = st.p0[None] * U0[:, None] - st.pz[None] * Uz[:, None]  # (S, Q)
    Cm = np.stack([pU, pU * st.p0[None], pU * st.pz[None]], axis=1)  # (S, 3, Q)
    lam = lam1.copy()
    scale = np.abs(M128_landau).max(1) + 1e-300
    for _ in range(100):
        f = fit1.f(lam)
        F = np.einsum("saq,sq->sa", Cm, f) - M128_landau
        if (np.abs(F).max(1) / scale < 1e-10).all():
            break
        J = np.einsum("saq,sq,bq->sab", Cm, f, fit1.phi)
        step = np.linalg.solve(J, F[..., None])[..., 0]
        n0 = np.linalg.norm(F, axis=1)
        accepted = np.zeros(len(lam), bool)
        for alpha in 0.5 ** np.arange(0, 30):
            todo = ~accepted
            if not todo.any():
                break
            trial = lam[todo] - alpha * step[todo]
            Ft = np.einsum("saq,sq->sa", Cm[todo], fit1.f(trial)) - M128_landau[todo]
            ok = np.linalg.norm(Ft, axis=1) < n0[todo] * (1 - 1e-4 * alpha) + 1e-300
            sub = np.flatnonzero(todo)[ok]
            lam[sub] = trial[ok]
            accepted[sub] = True
    f = fit1.f(lam)
    F = np.einsum("saq,sq->sa", Cm, f) - M128_landau
    return f, np.abs(F).max(1) / scale < 1e-8


def lp_subsample(T, noneq, n, seed):
    """Sites spread over temperature and both site groups, for the (slow) LP check."""
    rng = np.random.default_rng(seed + 1)
    order = np.argsort(T + 10 * noneq)
    pick = np.linspace(0, len(order) - 1, min(n, len(order))).astype(int)
    return np.sort(np.unique(order[pick] if n < len(order) else rng.permutation(len(order))))


def lp_realizable(A_nodes, M_ref, lead, K, idx):
    """True where some f >= 0 on the nodes matches every reference moment of degree <= K."""
    from scipy.optimize import linprog
    sel = np.flatnonzero(DEG <= K)
    ok = np.zeros(idx.size, bool)
    for j, s in enumerate(idx):
        sc = lead[DEG[sel], s]
        r = linprog(np.zeros(A_nodes.shape[1]), A_eq=A_nodes[sel] / sc[:, None],
                    b_eq=M_ref[sel, s] / sc, bounds=[(0, None)] * A_nodes.shape[1], method="highs")
        ok[j] = r.status == 0
    return ok


def poly_feq(st, rho, T, U0, Uz):
    """Alessandro's order-3 FEQ polynomial at the 16 nodes (Ux = Uy = 0, px = p_perp)."""
    import torch
    from quadrature import compute_feq_expression
    P = lambda a: torch.tensor(a, dtype=torch.float64).view(-1, 1)
    S = lambda a: torch.tensor(a, dtype=torch.float64).view(1, -1)
    z = S(np.zeros_like(rho))
    expr = compute_feq_expression(P(st.p0), P(st.p_perp), P(np.zeros(st.Q)), P(st.pz),
                                  S(U0), z, z, S(Uz), S(T)).numpy()
    return st.w[:, None] * rho[None] * expr


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", default=os.path.join(RLBM_DIR, "neurde_data", "rlbm_shock_tube_500.h5"))
    ap.add_argument("--steps", type=int, default=20, help="evenly spaced time steps to sample")
    ap.add_argument("--max-eq", type=int, default=500, help="near-equilibrium sites per step")
    ap.add_argument("--max-noneq", type=int, default=1500, help="non-equilibrium sites per step")
    ap.add_argument("--dirs", default="2,5;3,8;5,13;1,3",
                    help="candidate direction hop pairs, ';'-separated")
    ap.add_argument("--shells", type=int, nargs="+", default=[4], help="energy shell counts to try")
    ap.add_argument("--shell-scales", type=float, nargs="+", default=[1.0, 0.75, 0.6],
                    help="energy-grid reference temperatures to try (applied to the first --dirs pair)")
    ap.add_argument("--lp-sites", type=int, default=200, help="sites for the LP realizability check")
    ap.add_argument("--kmax-fit", type=int, default=4, help="highest matched degree K")
    ap.add_argument("--target-floor", type=float, default=0.02, help="pass threshold for degree-3 floor")
    ap.add_argument("--quick", action="store_true", help="3 steps, 200+300 sites per step")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.steps, args.max_eq, args.max_noneq = 3, 200, 300

    t0 = time.time()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    s128 = load_stencil128()
    A128 = monomials(s128["p0"], s128["pz"])
    data = load_samples(args.h5, args.steps, args.max_eq, args.max_noneq, args.seed)
    feq, f = data["feq"], data["f"]
    noneq = data["noneq"]
    groups = {"near-eq": ~noneq, "non-eq": noneq}
    S = feq.shape[1]
    M128 = A128 @ feq                              # (n_monomials, S)
    Mneq = A128 @ (f - feq)
    # normalize by sum |feq| p0^k: the 128 feq has negative high-shell populations at
    # cold sites, so the signed sum feq p0^k can be ~0 at degree >= 4
    lead = np.stack([(np.abs(feq) * s128["p0"][:, None] ** k).sum(0) for k in range(KMAX + 1)])

    out("# Phase 0: higher-moment budget (reference = Alessandro's 128-velocity data)")
    out()
    out(f"Data: `{os.path.relpath(args.h5, HERE)}`, steps {list(map(int, data['steps']))}")
    out(f"Sites sampled: {S} ({(~noneq).sum()} near-eq, {noneq.sum()} non-eq; "
        f"non-eq means sum|f-feq|/sum|feq| > {NONEQ_THRESHOLD:g}).")
    out("Moments are M(k,b) = sum_i g_i p0^(k-b) pz^b; errors are relative to "
        "sum_i |feq_i| p0^k (same degree), worst monomial of each degree.")
    summary = {"sites": int(S), "near_eq": int((~noneq).sum()), "non_eq": int(noneq.sum())}

    # ------------------------------------------------------------------ 0a
    out()
    out("## 0a. Moments present in the 128 data")
    out()
    out("Median (max) over sites of |M(k,b)| / sum |feq| p0^k, worst b per degree k. The last column is the share of sum |feq| p0^k coming from negative feq populations.")
    out()
    out("| degree k | feq, near-eq | feq, non-eq | f-feq, non-eq | negative-feq share (all sites) |")
    out("|---|---|---|---|---|")
    rel_eq = np.abs(M128) / lead[DEG]
    rel_neq = np.abs(Mneq) / lead[DEG]
    summary["0a"] = {}
    for k in range(KMAX + 1):
        a = np.stack([rel_eq[DEG == k][b] for b in range(k + 1)])
        c = np.stack([rel_neq[DEG == k][b] for b in range(k + 1)])
        s1, s2, s3 = stats(a.max(0), groups["near-eq"]), stats(a.max(0), noneq), stats(c.max(0), noneq)
        negs = (np.where(feq < 0, -feq, 0) * s128["p0"][:, None] ** k).sum(0) / lead[k]
        s4 = stats(negs, np.ones(S, bool))
        out(f"| {k} | {fmt(s1['median'])} ({fmt(s1['max'])}) | {fmt(s2['median'])} ({fmt(s2['max'])}) "
            f"| {fmt(s3['median'])} ({fmt(s3['max'])}) | {fmt(s4['median'])} ({fmt(s4['max'])}) |")
        summary["0a"][k] = {"feq_neareq": s1, "feq_noneq": s2, "neq_noneq": s3, "neg_share": s4}
    out()
    out("The f-feq column is the non-equilibrium content the solver carries; it is not "
        "part of the f_eq target, but it is what the 16-node f must hold during a rollout.")

    # ------------------------------------------------------------------ 0b
    out()
    out("## 0b. Which energy shells carry each moment (128 feq, averaged over sites)")
    out()
    shells = np.unique(np.round(s128["p0"], 8))
    sh_idx = np.searchsorted(shells, np.round(s128["p0"], 8))
    hot = data["T"] > 0.75
    summary["0b"] = {}
    for label, mask in (("T > 0.75", hot), ("T <= 0.75", ~hot)):
        if not mask.any():
            continue
        g = feq[:, mask]
        l2 = np.stack([(g[sh_idx == j] ** 2).sum(0) for j in range(4)])
        l2 = (l2 / l2.sum(0)).mean(1)
        out(f"**Sites with {label}** ({mask.sum()} sites)")
        out()
        out("| shell p0 | share of relative-L2 loss on f | " +
            " | ".join(f"share of sum abs(f) p0^{k}" for k in range(KMAX + 1)) + " |")
        out("|---" * (KMAX + 3) + "|")
        mom = []
        for k in range(KMAX + 1):
            m = np.stack([(np.abs(g[sh_idx == j]) * s128["p0"][sh_idx == j, None] ** k).sum(0) for j in range(4)])
            mom.append((m / m.sum(0)).mean(1))
        for j in range(4):
            out(f"| {shells[j]:.3f} | {l2[j]:.1%} | " +
                " | ".join(f"{mom[k][j]:.1%}" for k in range(KMAX + 1)) + " |")
        out()
        summary["0b"][label] = {"l2": l2.tolist(), "moment": [m.tolist() for m in mom]}
    out("A plain relative-L2 loss on f weights each shell like the L2 column; the moment "
        "columns show where each moment order actually lives.")

    # ------------------------------------------------------------------ 0c
    out()
    out("## 0c. Ceiling: best 16-node f per candidate stencil")
    out()
    out("Entropic K: f_i = w_i exp(lambda . phi_i) matching every 128-feq moment of degree <= K "
        "(only monomials independent on the nodes). Landau-3: the earlier plan's 3 constraints. "
        "Poly: Alessandro's FEQ polynomial at the nodes. Cells: median / p95 of the error at "
        "each degree; matched degrees are ~0 when the fit converged.")
    out()
    out("Realizable (LP): share of a stratified site subsample where SOME f >= 0 on the nodes "
        "matches the 128 feq moments of degree <= K (scipy linprog feasibility). If this is "
        "below 100%, no positive output, network or not, can carry those moments there.")
    cand = []
    dir_pairs = [tuple(int(x) for x in d.split(",")) for d in args.dirs.split(";")]
    for n_sh in args.shells:
        for na, nb in dir_pairs:
            try:
                cand.append(Stencil16((na, nb), n_shells=n_sh))
            except ValueError as e:
                out(f"- skipped ({na},{nb}) with {n_sh} shells: {e}")
    for sc in args.shell_scales:
        if sc != 1.0:
            cand.append(Stencil16(dir_pairs[0], n_shells=args.shells[0], shell_scale=sc))
    lp_idx = lp_subsample(data["T"], noneq, args.lp_sites, args.seed)
    lp128 = {K: lp_realizable(A128, M128, lead, K, lp_idx) for K in (2, 3)}
    out()
    out(f"Reference, all 128 nodes: realizable K=2 {lp128[2].mean():.0%}, K=3 {lp128[3].mean():.0%} "
        f"of {lp_idx.size} sites.")
    summary["lp128"] = {K: float(v.mean()) for K, v in lp128.items()}
    summary["0c"] = {}
    for st in cand:
        out()
        out(f"### {st.name}  (Q = {st.Q}, v0 = {st.v0:.4f})")
        A16 = monomials(st.p0, st.pz)
        lp16 = {K: lp_realizable(A16, M128, lead, K, lp_idx) for K in (2, 3)}
        T_lp = data["T"][lp_idx]
        out()
        out(f"Realizable (LP): K=2 {lp16[2].mean():.0%}, K=3 {lp16[3].mean():.0%}"
            + (f"; K=3 fails for T in [{T_lp[~lp16[3]].min():.2f}, {T_lp[~lp16[3]].max():.2f}]"
               if (~lp16[3]).any() else "") + ".")
        A16 = monomials(st.p0, st.pz)
        rows, res_c = [], {}
        lam_prev = None
        fit1 = lam1 = None
        for K in range(1, args.kmax_fit + 1):
            fit = EntropicFit(st, K)
            lam0 = np.zeros((S, fit.r))
            if lam_prev is None:
                lam0[:, 0] = np.log(np.maximum((fit.C @ M128[fit.sel])[0], 1e-300))
            else:
                lam0[:, :lam_prev.shape[1]] = lam_prev
            M_sel = M128[fit.sel]
            lead_sel = lead[DEG[fit.sel]].T
            lam, conv = fit.solve(M_sel, lead_sel, lam0)
            lam_prev = lam
            if K == 1:
                fit1, lam1 = fit, lam
            err = degree_error(A16 @ fit.f(lam).T, M128, lead)
            rows.append((f"entropic K={K} ({fit.r} moments)", err, conv, None))
        U0, Uz = data["U0"], data["Uz"]
        pU128 = s128["p0"][:, None] * U0[None] - s128["pz"][:, None] * Uz[None]
        M_landau = np.stack([(pU128 * feq).sum(0), (pU128 * s128["p0"][:, None] * feq).sum(0),
                             (pU128 * s128["pz"][:, None] * feq).sum(0)], axis=1)
        fL, convL = landau_fit(st, fit1, lam1, U0, Uz, M_landau)
        rows.append(("Landau-3 (earlier plan)", degree_error(A16 @ fL.T, M128, lead), convL, None))
        if st.shell_scale == 1.0:  # the FEQ polynomial is built for Alessandro's shells only
            fP = poly_feq(st, data["rho"], data["T"], U0, Uz)
            rows.append(("Poly FEQ at nodes", degree_error(A16 @ fP, M128, lead), np.ones(S, bool),
                         float((fP < 0).mean())))

        for gname, gmask in list(groups.items()) + [("all", np.ones(S, bool))]:
            if not gmask.any():
                continue
            out()
            out(f"{gname} sites:")
            out()
            out("| target | converged | " + " | ".join(f"deg {k}" for k in range(KMAX + 1)) + " |")
            out("|---" * (KMAX + 3) + "|")
            for name, err, conv, negfrac in rows:
                cells = []
                for k in range(KMAX + 1):
                    s = stats(err[k], gmask & conv)
                    cells.append(f"{fmt(s['median'])} / {fmt(s['p95'])}")
                extra = f" (neg f: {negfrac:.0%})" if negfrac is not None else ""
                out(f"| {name}{extra} | {conv[gmask].mean():.1%} | " + " | ".join(cells) + " |")
                res_c.setdefault(name, {})[gname] = {
                    "converged": float(conv[gmask].mean()),
                    "deg": {k: stats(err[k], gmask & conv) for k in range(KMAX + 1)},
                }
        res_c["lp"] = {K: float(v.mean()) for K, v in lp16.items()}
        summary["0c"][st.name] = res_c

    # ------------------------------------------------------------------ 0d
    out()
    out("## 0d. Recommendation")
    out()
    out("Per candidate: highest degree K whose positive entropic fit converges at >= 99% of all "
        "sites, then the p95 (all sites) error at degree 3 (the floor) and at degree K+1.")
    out()
    best = None
    for st in cand:
        res = summary["0c"][st.name]
        kfeas = 0
        for K in range(1, args.kmax_fit + 1):
            key = next(n for n in res if n.startswith(f"entropic K={K} "))
            if res[key]["all"]["converged"] >= 0.99:
                kfeas = K
        if kfeas == 0:
            out(f"- {st.name}: no K converges at >= 99% of sites.")
            continue
        key = next(n for n in res if n.startswith(f"entropic K={kfeas} "))
        nxt = min(kfeas + 1, KMAX)
        floor3 = res[key]["all"]["deg"][3]["p95"]
        nxt_err = res[key]["all"]["deg"][nxt]["p95"]
        nxt_txt = f", at degree {nxt} = {fmt(nxt_err)}" if nxt != 3 else ""
        out(f"- {st.name} (Q={st.Q}): K = {kfeas}; p95 error at degree 3 = {fmt(floor3)}{nxt_txt}; "
            f"LP-realizable K=3 at {res['lp'][3]:.0%} of sites.")
        score = (-kfeas, np.nan_to_num(floor3, nan=np.inf), np.nan_to_num(nxt_err, nan=np.inf))
        if st.Q == 16 and (best is None or score < best[0]):
            best = (score, st, kfeas, floor3)
    out()
    if best is not None:
        _, st, kfeas, floor3 = best
        ok = np.isfinite(floor3) and floor3 < args.target_floor
        out(f"**Recommended:** stencil {st.name}, Stage 1 target = entropic f_eq matching all "
            f"moments of degree <= {kfeas}.")
        out(f"Degree-3 floor (p95, all sites) = {fmt(floor3)} -> "
            f"{'PASS' if ok else 'FAIL'} against the {args.target_floor:.0%} threshold.")
        out("Loss: moment-space terms for each degree k <= 5, each normalized by its own 128 "
            "moment (equal footing per order, see 0b), plus a small f-space term.")
        summary["0d"] = {"stencil": st.name, "K": kfeas, "floor_deg3": floor3, "pass": bool(ok)}
    else:
        out("No 16-node candidate passed the convergence test; see the tables above.")
        summary["0d"] = None
    out()
    out(f"_Runtime {time.time() - t0:.0f} s._")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "phase0_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "phase0_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"\nWrote {os.path.join(args.out, 'phase0_report.md')}")


if __name__ == "__main__":
    main()
