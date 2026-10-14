"""
Evaluate full rollouts on the 16-node set against the 128-velocity run
(TRAINING_PLAN.md, V4/V5).

Runs from the stored 16-node state at t = 0 (stage2.h5) up to the last stored time of
the 128 run with
  * the trained network (--checkpoint), and/or
  * the classical equilibrium (--classical): exact U from the moments + Alessandro's
    FEQ polynomial, i.e. the RLBM_STENCIL=16 run of relativistic_hydrodynamics,
    started from the same state; this is the baseline the network must match or beat.
Reports, at each requested time, relative L2 and max errors of n, eps, T, P, Uz
against the 128 run, plus the drift of the conserved totals; writes overlay plots
and an .npz with the profiles.

Usage: python eval.py --checkpoint checkpoints/stage2/best.pt --classical
"""

import argparse
import math
import os

import numpy as np
import torch
import yaml

import rlbm1d as R
from model import NeurDE1D
from train_stage_2 import Stage2Data, step_network

FIELDS = ("n", "eps", "T", "P", "Uz")


def derived(n, Uz, T):
    return {"n": n, "eps": 3 * n * T, "T": T, "P": n * T, "Uz": Uz}


@torch.no_grad()
def rollout(step_fn, f, st, n_steps):
    """Fields after every step (index 0 = start) and totals of N0, T00, T0z."""
    out = {k: [] for k in ("n", "Uz", "T")}
    totals = []
    for s in range(n_steps + 1):
        mom = R.moments(f, st)
        _, Uz, n, T = R.landau(*mom)
        for k, v in (("n", n), ("Uz", Uz), ("T", T)):
            out[k].append(v)
        totals.append(torch.stack([mom[0].sum(), mom[2].sum(), mom[3].sum()]))
        if s < n_steps:
            f = step_fn(f)
    return {k: torch.stack(v) for k, v in out.items()}, torch.stack(totals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--params", default=os.path.join(R.HERE, "params.yml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--times", type=float, nargs="+", default=[0.8, 1.6, 2.4, 3.19],
                    help="fm/c; the 128 run is stored up to 499 * dt_128 = 3.195")
    ap.add_argument("--out", default=os.path.join(R.HERE, "results", "eval"))
    args = ap.parse_args()
    if not (args.checkpoint or args.classical):
        raise SystemExit("give --checkpoint and/or --classical")
    cfg = yaml.safe_load(open(args.params))
    dtype = getattr(torch, cfg["dtype"])
    st = R.Stencil1D(cfg["stencil"]["hops"], cfg["stencil"]["T0"], device=args.device, dtype=dtype)
    data = Stage2Data(os.path.join(R.HERE, cfg["data_dir"], "stage2.h5"), args.device, dtype)
    k0 = int(np.flatnonzero(data.init_steps == 0)[0])
    t_max = (data.n_steps - 1) * R.DT_128
    times = [t for t in args.times if t <= t_max + 1e-9]
    n_steps = math.ceil(max(times) / st.dt)

    runs = {}
    if args.classical:
        def classical(f):
            mom = R.moments(f, st)
            U0, Uz, n, T = R.landau(*mom)
            feq = R.poly_feq(st, n, U0, Uz, T)
            return R.stream(R.collide(f, feq, U0, Uz, R.tau(n, T, st.v0), st), st)
        runs["classical"] = rollout(classical, data.f_init[k0], st, n_steps)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model = NeurDE1D(**ckpt["cfg"]["model"]).to(args.device, dtype)
        model.load_state_dict(ckpt["model"])
        model.eval()
        runs["network"] = rollout(lambda f: step_network(model, f, st), data.f_init[k0], st, n_steps)

    os.makedirs(args.out, exist_ok=True)
    saved = {}
    print(f"{st.name}: {n_steps} steps of dt = {st.dt:.6f} vs the 128 run (dt = {R.DT_128:.6f})")
    for name, (_, tot) in runs.items():
        drift = ((tot - tot[0]).abs().amax(0) / tot[0, [0, 1, 1]].abs()).tolist()
        print(f"  {name:9s} drift of total N0, T00, T0z (rel. to N0, T00, T00): "
              + ", ".join(f"{x:.1e}" for x in drift))
    for t in times:
        ref = derived(*(data.target(k, t / R.DT_128) for k in ("n", "Uz", "T")))
        print(f"\n== t = {t:g} fm/c" + "".join(f"   {k:>17s}" for k in FIELDS))
        print(" " * 18 + "".join(f"   {'relL2 / max':>17s}" for _ in FIELDS))
        profiles = {"ref": ref}
        for name, (fields, _) in runs.items():
            prof = derived(*(R.interp_time(fields[k], t / st.dt) for k in ("n", "Uz", "T")))
            profiles[name] = prof
            cells = []
            for k in FIELDS:
                d = prof[k] - ref[k]
                cells.append(f"{R.relL2(prof[k], ref[k]):.1e} / {(d.abs().max() / ref[k].abs().max()):.1e}")
            print(f"  {name:14s}" + "".join(f"   {c:>17s}" for c in cells))
        for name, prof in profiles.items():
            for k in FIELDS:
                saved[f"t{t:g}_{name}_{k}"] = prof[k].cpu().numpy()

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.arange(ref["n"].numel()) - ref["n"].numel() // 2
        fig, axes = plt.subplots(len(FIELDS), 1, figsize=(10, 2.4 * len(FIELDS)), sharex=True)
        for ax, k in zip(axes, FIELDS):
            ax.plot(z, ref[k].cpu(), "k-", lw=2, label="128 run")
            for name in runs:
                ax.plot(z, profiles[name][k].cpu(), "--", lw=1.2, label=f"{st.Q}-node {name}")
            ax.set_ylabel(k)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"t = {t:g} fm/c")
        axes[-1].set_xlabel("z (cells)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"eval_t{t:.3f}.png"), dpi=130)
        plt.close(fig)
    np.savez_compressed(os.path.join(args.out, "profiles.npz"), **saved)
    print(f"\nplots and profiles in {args.out}")


if __name__ == "__main__":
    main()
