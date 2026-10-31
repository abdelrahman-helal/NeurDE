"""
Print the moment matrices N^a and T^ab of the 128-velocity run and of the 16-node set
at two sites, side by side with exact Maxwell-Juttner.

At one 128-run step and two sites (default: the flowing equilibrium site with the
largest |Uz|, and the most non-equilibrium site, i.e. a shock front), it prints:
  128 f        the 128 run's state
  128 feq      Alessandro's equilibrium on the 128 set
  MJ exact     N = nU, T = 4nT UU - nT eta, from the 128 run's own n, U, T
  16 run       the classical 16-node run (stage2.h5 start + step_classical) at the
               same physical time (linear interpolation between its two nearest steps)
  16 poly      Alessandro's FEQ polynomial on the 16 nodes, with the 128 run's n, U, T
  16 entropic  the Stage 1 target: positive fit to the 128 feq moments of degree <= 2
The 16-node set has px = py = 0 (each node is a ring around z), so its transverse
components are N^x = N^y = 0 and T^xx = T^yy = sum f p_perp^2 / 2.
Also prints the degree-3 z moments sum f p0^(3-b) pz^b, which carry the 128 run's
extra accuracy, and saves everything to an .npz.

Usage: python inspect_moments.py [--step 300] [--sites Z1 Z2] [--h5 PATH]
"""

import argparse
import math
import os

os.environ["RLBM_STENCIL"] = "128"  # stencil.Stencil (128 set) needs config's NPOP = 128

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import rlbm1d as R  # noqa: E402
from stencil import Stencil  # noqa: E402  (relativistic_hydrodynamics)
from train_stage_2 import Stage2Data, step_classical  # noqa: E402

NAMES = ("t", "x", "y", "z")
KB3 = [(3, b) for b in range(4)]  # p0^3, p0^2 pz, p0 pz^2, pz^3


def tensors_3d(f, P):
    """N^a (4,) and T^ab (4, 4) of populations f (Q,) with four-momenta P (Q, 4)."""
    return P.T @ f, (P * f[:, None]).T @ P


def tensors_1d(f, st):
    """Same on a reduced set: rings around z, so N^x = N^y = 0, T^xx = T^yy = sum f p_perp^2 / 2."""
    p0, pz, pp = st.np["p0"], st.np["pz"], st.np["p_perp"]
    N = np.array([f @ p0, 0.0, 0.0, f @ pz])
    T = np.zeros((4, 4))
    T[0, 0], T[0, 3], T[3, 3] = f @ (p0 * p0), f @ (p0 * pz), f @ (pz * pz)
    T[3, 0] = T[0, 3]
    T[1, 1] = T[2, 2] = 0.5 * f @ (pp * pp)
    return N, T


def mj_tensors(n, U, T):
    """Exact massless MJ: N = nU, T^ab = (eps + P) U^a U^b - P eta^ab with eps = 3nT, P = nT."""
    eta = np.diag([1.0, -1.0, -1.0, -1.0])
    return n * U, 4 * n * T * np.outer(U, U) - n * T * eta


def deg3(f, p0, pz):
    return np.array([f @ (p0 ** (k - b) * pz ** b) for k, b in KB3])


def show(rows, title):
    """Side-by-side blocks, one per label: the row N^a, then the four rows of T^ab."""
    cell = lambda v: f"{v:+.4e}".rjust(12)
    block = lambda vec: "".join(cell(v) for v in vec) + "   "
    labels = list(rows)
    print(f"\n  {title}")
    print("          " + "".join(l.ljust(51) for l in labels))
    print("          " + "".join("".join(f"{c:>12s}" for c in NAMES) + "   " for _ in labels))
    print("    N^a   " + "".join(block(rows[l][0]) for l in labels))
    for a in range(4):
        print(f"    T^{NAMES[a]}b  " + "".join(block(rows[l][1][a]) for l in labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", default=os.path.join(R.RLBM_DIR, "neurde_data", "rlbm_shock_tube_500.h5"))
    ap.add_argument("--params", default=os.path.join(R.HERE, "params.yml"))
    ap.add_argument("--step", type=int, default=300, help="128-run step (time = step * dt_128)")
    ap.add_argument("--sites", type=int, nargs=2, default=None, help="grid indices z (0..NZ-1)")
    ap.add_argument("--out", default=os.path.join(R.HERE, "results", "moments_inspect.npz"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.params))
    dtype = getattr(torch, cfg["dtype"])
    np.set_printoptions(linewidth=200)

    # ---------------------------------------------------------------- 128 run
    s128 = Stencil()
    P = np.stack([s128.p0.cpu().numpy(), s128.px.cpu().numpy(), s128.py.cpu().numpy(), s128.pz.cpu().numpy()], 1)
    with h5py.File(args.h5, "r") as h:
        t = args.step
        f = h["f"][t].reshape(128, -1)
        feq = h["feq"][t].reshape(128, -1)
        N_stored = h["N_moments"][t].reshape(4, -1)
        fld = {k: h[k][t].reshape(-1) for k in ("rho", "U0", "Ux", "Uy", "Uz", "T")}
    nz = f.shape[1]
    noneq = np.abs(f - feq).sum(0) / np.abs(feq).sum(0)
    if args.sites is None:
        eq = noneq < 1e-6
        site_a = int(np.flatnonzero(eq)[np.argmax(np.abs(fld["Uz"][eq]))])
        site_b = int(np.argmax(noneq))
        sites = [site_a, site_b]
    else:
        sites = args.sites
    print(f"128 run, step {t} (t = {t * R.DT_128:.4f} fm/c); N^a from f vs stored N_moments: "
          f"max |diff| = {np.abs(P.T @ f - N_stored).max():.1e}")

    # ---------------------------------------------------------------- 16-node set
    st = R.Stencil1D(cfg["stencil"]["hops"], cfg["stencil"]["T0"], device="cpu", dtype=dtype)
    data = Stage2Data(os.path.join(R.HERE, cfg["data_dir"], "stage2.h5"), "cpu", dtype)
    k0 = int(np.flatnonzero(data.init_steps == 0)[0])
    s_exact = t / data.ratio                       # 16-node steps to reach the same physical time
    s_lo = int(math.floor(s_exact))
    f16 = data.f_init[k0]
    with torch.no_grad():
        for _ in range(s_lo):
            f16 = step_classical(f16, st)
        f16_next = step_classical(f16, st)
    a = s_exact - s_lo
    f16_run = ((1 - a) * f16 + a * f16_next).cpu().numpy()
    print(f"16-node classical run: {s_exact:.3f} steps of dt = {st.dt:.6f} "
          f"(interpolated between steps {s_lo} and {s_lo + 1})")

    kb2 = R.monomial_list(2)
    fit = R.EntropicFit(st.np["p0"], st.np["pz"], st.np["w"], kb2)
    A128_2 = R.monomials(P[:, 0], P[:, 3], kb2)
    lead = np.stack([(np.abs(feq[:, sites]) * P[:, :1] ** k).sum(0) for k in range(3)])
    f16_ent, _, conv = fit.solve(A128_2 @ feq[:, sites], lead[[k for k, _ in kb2]])
    tt = lambda x: torch.tensor(x[sites], dtype=dtype)
    f16_poly = R.poly_feq(st, tt(fld["rho"]), tt(fld["U0"]), tt(fld["Uz"]), tt(fld["T"])).cpu().numpy()

    saved = {}
    for j, z in enumerate(sites):
        U = np.array([fld["U0"][z], fld["Ux"][z], fld["Uy"][z], fld["Uz"][z]])
        n, T = fld["rho"][z], fld["T"][z]
        kind = "near equilibrium" if noneq[z] < 1e-6 else "non-equilibrium"
        print(f"\n=== site z = {z} ({z - nz // 2:+d} from centre), {kind}: sum|f - feq|/sum|feq| = {noneq[z]:.2e}")
        print(f"    128 run: n = {n:.6f}, T = {T:.6f}, U = ({U[0]:.6f}, {U[1]:+.2e}, {U[2]:+.2e}, {U[3]:+.6f})")
        rows = {"128 f": tensors_3d(f[:, z], P), "128 feq": tensors_3d(feq[:, z], P), "MJ exact": mj_tensors(n, U, T)}
        show(rows, "128-velocity set")
        rows16 = {"16 run (same time)": tensors_1d(f16_run[:, z], st), "16 poly f^eq": tensors_1d(f16_poly[:, j], st),
                  f"16 entropic (conv={bool(conv[j])})": tensors_1d(f16_ent[:, j], st)}
        show(rows16, "16-node set")

        mj3 = R.mj_moments(torch.tensor([n]), torch.tensor([U[0]]), torch.tensor([U[3]]), torch.tensor([T]),
                           KB3)[:, 0].numpy()
        d3 = {"128 f": deg3(f[:, z], P[:, 0], P[:, 3]), "128 feq": deg3(feq[:, z], P[:, 0], P[:, 3]), "MJ exact": mj3,
              "16 run": deg3(f16_run[:, z], st.np["p0"], st.np["pz"]),
              "16 poly": deg3(f16_poly[:, j], st.np["p0"], st.np["pz"]),
              "16 entropic": deg3(f16_ent[:, j], st.np["p0"], st.np["pz"])}
        print("\n  degree-3 z moments" + "".join(f"{l:>14s}" for l in d3))
        for i, (k, b) in enumerate(KB3):
            print(f"    p0^{k - b} pz^{b}        " + "".join(f"{v[i]:+14.5e}" for v in d3.values()))

        for name, (Nv, Tm) in {**rows, **rows16}.items():
            key = name.split(" (")[0].replace(" ", "_").replace("^", "")
            saved[f"z{z}_{key}_N"], saved[f"z{z}_{key}_T"] = Nv, Tm
        for name, v in d3.items():
            saved[f"z{z}_{name.replace(' ', '_')}_deg3"] = v
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, sites=np.array(sites), step=t, **saved)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
