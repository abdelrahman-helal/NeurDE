"""
Check that the torch 1D solver in rlbm1d.py, run classically (exact U from moments +
Alessandro's FEQ polynomial, no network), reproduces relativistic_hydrodynamics run
with RLBM_STENCIL=16 step for step. Also checks that the conservation projection
leaves an already-conserving f_eq unchanged, and that totals are conserved.

Usage:  python test_solver1d.py [--steps 20] [--hops 2 5] [--T0 0.75]
"""

import argparse
import os

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--hops", type=int, nargs="+", default=[2, 5])
ap.add_argument("--T0", type=float, default=0.75)
args = ap.parse_args()

# Select the matching reduced set in Alessandro's code before anything imports config
os.environ["RLBM_STENCIL"] = {(2, 5): "16", (1, 3, 5): "24"}[tuple(args.hops)]
os.environ["RLBM_T0"] = str(args.T0)

import torch  # noqa: E402

import rlbm1d as R  # noqa: E402
import collide as rlbm_collide  # noqa: E402  (relativistic_hydrodynamics)
from init import initialize  # noqa: E402
from propagate import propagate  # noqa: E402
from stencil import make_stencil  # noqa: E402
from config import DEVICE  # noqa: E402


def main():
    ref_st = make_stencil()
    f_ref, param = initialize(ref_st, init_type="bamps")
    st = R.Stencil1D(args.hops, args.T0, device=DEVICE)
    f = f_ref.reshape(st.Q, -1).clone()
    start = torch.stack([m.sum() for m in R.moments(f, st)])

    worst_f = worst_proj = 0.0
    for step in range(1, args.steps + 1):
        # Alessandro's code
        f2, param = rlbm_collide.collide(f_ref, ref_st, param)
        f_ref = propagate(f2, ref_st)
        # 1D solver, classical equilibrium
        mom = R.moments(f, st)
        U0, Uz, n, T = R.landau(*mom)
        feq = R.poly_feq(st, n, U0, Uz, T)
        feq_p = R.project_conservation(feq, U0, Uz, *mom, st)
        worst_proj = max(worst_proj, ((feq_p - feq).abs().max() / feq.abs().max()).item())
        f = R.stream(R.collide(f, feq, U0, Uz, R.tau(n, T, st.v0), st), st)
        worst_f = max(worst_f, ((f - f_ref.reshape(st.Q, -1)).abs().max() / f_ref.abs().max()).item())

    end = torch.stack([m.sum() for m in R.moments(f, st)])
    drift = ((end - start).abs() / start[[0, 0, 2, 2, 2]].abs()).tolist()
    print(f"{st.name}, {args.steps} steps")
    print(f"  max |f_1D - f_Alessandro| / max|f|        = {worst_f:.1e}   (expect ~1e-12 or below)")
    print(f"  max change from conservation projection    = {worst_proj:.1e}   (expect ~1e-12: poly f_eq already conserves)")
    print(f"  drift of totals N0, Nz, T00, T0z, Tzz      = " + ", ".join(f"{d:.1e}" for d in drift)
          + "   (N0, T00, T0z conserved; Nz, Tzz are not)")


if __name__ == "__main__":
    main()
