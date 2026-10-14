"""
Build the NeurDE training data from the 128-velocity run (TRAINING_PLAN.md, step 3).

Stage 1 (data/stage1.h5), one row per sampled site and step:
  moments  (S, 5)  N0, N3, T00, T03, T33 of the 128 run's f (the network input)
  Mref     (S, 10) the 128 run's f_eq moments of degree <= 3 (p0^(k-b) pz^b, code units)
  lead     (S, 4)  sum |feq| p0^k, k = 0..3 (scale of a degree-k moment)
  Feq16    (S, Q)  positive entropic f_eq on the reduced set matching Mref's degree <= 2
  lam, conv        its multipliers and Newton convergence flag
  vz, n, T         the 128 run's Uz/U0, density, temperature
  noneq, step      non-equilibrium flag, h5 step index (time = step * dt_128)
Sites: all non-equilibrium sites (sum|f-feq|/sum|feq| > 1e-6, capped) plus a random
sample of near-equilibrium sites per step.

Stage 2 (data/stage2.h5):
  f16_init (n_init, Q, NZ)  starting states on the reduced set: positive entropic fit
                            to the degree <= 2 moments of the 128 run's f (falls back
                            to f_eq's moments where the fit does not converge)
  init_steps                h5 step index of each starting state
  n, U0, Uz, T (n_steps, NZ) the 128 run's fields at every step (Stage 2 targets)

Usage: python make_dataset.py [--h5 PATH] [--out-dir data] [--hops 2 5] [--T0 0.75]
"""

import argparse
import os

os.environ["RLBM_STENCIL"] = "128"  # stencil.Stencil (128 set) needs config's NPOP = 128

import h5py  # noqa: E402
import numpy as np  # noqa: E402

import rlbm1d as R  # noqa: E402
from stencil import Stencil  # noqa: E402  (relativistic_hydrodynamics)

NONEQ_THRESHOLD = 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", default=os.path.join(R.RLBM_DIR, "neurde_data", "rlbm_shock_tube_500.h5"))
    ap.add_argument("--out-dir", default=os.path.join(R.HERE, "data"))
    ap.add_argument("--hops", type=int, nargs="+", default=[2, 5])
    ap.add_argument("--T0", type=float, default=0.75)
    ap.add_argument("--eq-per-step", type=int, default=500)
    ap.add_argument("--noneq-per-step", type=int, default=2000)
    ap.add_argument("--init-every", type=int, default=5, help="Stage 2 starting state every N steps")
    ap.add_argument("--max-steps", type=int, default=None, help="only the first N steps (testing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    s128 = Stencil()
    p0_128, pz_128 = s128.p0.cpu().numpy(), s128.pz.cpu().numpy()
    st = R.Stencil1D(args.hops, args.T0)
    kb3, kb2 = R.monomial_list(3), R.monomial_list(2)
    deg3 = np.array([k for k, _ in kb3])
    A128 = R.monomials(p0_128, pz_128, kb3)
    fit = R.EntropicFit(st.np["p0"], st.np["pz"], st.np["w"], kb2)
    lead_of = lambda g: np.stack([(np.abs(g) * p0_128[:, None] ** k).sum(0) for k in range(4)])

    rows = {k: [] for k in ("moments", "Mref", "lead", "vz", "n", "T", "noneq", "step")}
    inits, init_steps, init_conv = [], [], []
    with h5py.File(args.h5, "r") as h:
        n_steps = h["feq"].shape[0] if args.max_steps is None else args.max_steps
        fields = {k: h[src][:n_steps].reshape(n_steps, -1) for k, src in
                  (("n", "rho"), ("U0", "U0"), ("Uz", "Uz"), ("T", "T"))}
        for t in range(n_steps):
            feq = h["feq"][t].reshape(128, -1)
            f = h["f"][t].reshape(128, -1)
            N = h["N_moments"][t].reshape(4, -1)
            Tm = h["T_moments"][t].reshape(10, -1)  # T00 T01 T02 T03 T11 T12 T13 T22 T23 T33

            # Stage 1 sites
            r = np.abs(f - feq).sum(0) / np.abs(feq).sum(0)
            noneq_idx, eq_idx = np.flatnonzero(r > NONEQ_THRESHOLD), np.flatnonzero(r <= NONEQ_THRESHOLD)
            if noneq_idx.size > args.noneq_per_step:
                noneq_idx = rng.choice(noneq_idx, args.noneq_per_step, replace=False)
            if eq_idx.size > args.eq_per_step:
                eq_idx = rng.choice(eq_idx, args.eq_per_step, replace=False)
            idx = np.sort(np.concatenate([noneq_idx, eq_idx]))
            rows["moments"].append(np.stack([N[0], N[3], Tm[0], Tm[3], Tm[9]])[:, idx])
            rows["Mref"].append(A128 @ feq[:, idx])
            rows["lead"].append(lead_of(feq[:, idx]))
            rows["vz"].append(fields["Uz"][t, idx] / fields["U0"][t, idx])
            rows["n"].append(fields["n"][t, idx])
            rows["T"].append(fields["T"][t, idx])
            rows["noneq"].append(r[idx] > NONEQ_THRESHOLD)
            rows["step"].append(np.full(idx.size, t))

            # Stage 2 starting state on the reduced set
            if t % args.init_every == 0:
                M_f, lead_f = R.monomials(p0_128, pz_128, kb2) @ f, lead_of(f)
                f16, _, conv = fit.solve(M_f, lead_f[deg3[:6]])
                if not conv.all():  # fall back to the equilibrium's moments
                    bad = ~conv
                    M_e, lead_e = R.monomials(p0_128, pz_128, kb2) @ feq[:, bad], lead_of(feq[:, bad])
                    f16[:, bad], _, _ = fit.solve(M_e, lead_e[deg3[:6]])
                inits.append(f16)
                init_steps.append(t)
                init_conv.append(conv.mean())
            if t % 50 == 0:
                print(f"step {t}/{n_steps}: {idx.size} sites ({noneq_idx.size} non-eq)", flush=True)

    data = {k: np.concatenate(v, axis=-1) for k, v in rows.items()}
    S = data["n"].size
    print(f"Stage 1: {S} samples; fitting positive f_eq on the {st.Q}-node set...", flush=True)
    lead_rows = data["lead"][deg3[:6]]
    Feq16, lam, conv = fit.solve(data["Mref"][:6], lead_rows)
    M16 = R.monomials(st.np["p0"], st.np["pz"], kb3) @ Feq16
    err = np.abs(M16 - data["Mref"]) / data["lead"][deg3]
    print(f"  converged: {conv.mean():.4%}   min f_eq: {Feq16.min():.3e}")
    print(f"  degree <= 2 max error: {err[:6].max():.1e}   degree 3 error p95 / max: "
          f"{np.percentile(err[6:].max(0), 95):.2%} / {err[6:].max():.2%}")
    print(f"Stage 2: {len(inits)} starting states, fit converged on "
          f"{np.mean(init_conv):.4%} of sites (rest fall back to f_eq moments)")

    with h5py.File(os.path.join(args.out_dir, "stage1.h5"), "w") as o:
        for k in ("moments", "Mref", "lead"):
            o.create_dataset(k, data=data[k].T)
        o.create_dataset("Feq16", data=Feq16.T)
        o.create_dataset("lam", data=lam)
        o.create_dataset("conv", data=conv)
        for k in ("vz", "n", "T", "noneq", "step"):
            o.create_dataset(k, data=data[k])
        o.attrs.update(hops=st.hops, T0=st.T0, Q=st.Q, kb3=np.array(kb3), source=args.h5, dt128=R.DT_128)
    with h5py.File(os.path.join(args.out_dir, "stage2.h5"), "w") as o:
        o.create_dataset("f16_init", data=np.stack(inits))
        o.create_dataset("init_steps", data=np.array(init_steps))
        for k, v in fields.items():
            o.create_dataset(k, data=v)
        o.attrs.update(hops=st.hops, T0=st.T0, Q=st.Q, source=args.h5, dt128=R.DT_128, dt16=st.dt)
    print(f"Wrote {args.out_dir}/stage1.h5 and stage2.h5")


if __name__ == "__main__":
    main()
