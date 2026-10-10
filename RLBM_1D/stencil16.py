"""
Candidate 16-node velocity sets for the 1D (z-only) relativistic shock tube.

A node is a pair (energy shell j, direction k):

  * Energy shells: the Gauss rule for the massless measure p0 e^{-p0} dp0, i.e. the
    roots of the generalized Laguerre polynomial L_n^(1). For n = 4 these are exactly
    the four p0 values of Alessandro's 128-velocity stencil.
  * Directions: two symmetric pairs mu = cos(theta) = +-mu_a, +-mu_b. Their weights
    reproduce the isotropic averages <1> = 1, <mu^2> = 1/3, <mu^4> = 1/5. The integer
    hops (n_a, n_b) fix the last free parameter: mu = nz * v0, so every particle lands
    on a grid node after one step and streaming stays exact.

Node weight = W_shell(j) * W_dir(k), with sum = 1 (same normalization as stencil.py).

Run directly to print the node table and its exactness:
    python stencil16.py
"""

import math

import numpy as np
from scipy.special import roots_genlaguerre


def laguerre_shells(n_shells=4):
    """Energy shells p0_j and normalized weights W_j for the measure p0 e^{-p0} dp0."""
    p0, W = roots_genlaguerre(n_shells, 1.0)
    return p0, W / W.sum()


def solve_direction_rule(n_a, n_b):
    """Direction pair (+-mu_a, +-mu_b) with mu = n * v0 matching <1>, <mu^2>, <mu^4>.

    Eliminating the two weights from
        W_a + W_b = 1,  W_a mu_a^2 + W_b mu_b^2 = 1/3,  W_a mu_a^4 + W_b mu_b^4 = 1/5
    with mu_b = (n_b/n_a) mu_a leaves  r x^2 - (1 + r) x / 3 + 1/5 = 0,
    where x = mu_a^2 and r = (n_b/n_a)^2.
    """
    r = (n_b / n_a) ** 2
    disc = ((1 + r) / 3) ** 2 - 4 * r / 5
    if disc < 0:
        raise ValueError(f"no real direction rule for hops ({n_a}, {n_b})")
    for x in sorted([((1 + r) / 3 - math.sqrt(disc)) / (2 * r),
                     ((1 + r) / 3 + math.sqrt(disc)) / (2 * r)]):
        mu_a = math.sqrt(x)
        mu_b = mu_a * n_b / n_a
        if x <= 0 or mu_b > 1:
            continue
        W_a = (1 / 3 - mu_b ** 2) / (mu_a ** 2 - mu_b ** 2)
        W_b = 1 - W_a
        if W_a > 0 and W_b > 0:
            return dict(n_a=n_a, n_b=n_b, v0=mu_a / n_a, mu_a=mu_a, mu_b=mu_b, W_a=W_a, W_b=W_b)
    raise ValueError(f"no valid direction rule for hops ({n_a}, {n_b}) (needs mu_b <= 1, weights > 0)")


class Stencil16:
    """Product stencil: Laguerre energy shells x two symmetric lattice directions.

    Attributes mirror relativistic_hydrodynamics/stencil.py where they exist
    (p0, pz, w, nz, shifts_list); px/py are replaced by p_perp because each node
    stands for a ring of directions around the z axis.
    """

    def __init__(self, dirs=(2, 5), n_shells=4, shell_scale=1.0):
        """shell_scale s rescales the energy shells to the Gauss rule for p0 e^{-p0/s}
        (s = reference temperature of the energy grid; 1.0 = Alessandro's shells).
        Massless particles all move at c, so s does not change streaming."""
        n_a, n_b = dirs
        p0_s, W_s = laguerre_shells(n_shells)
        p0_s = p0_s * shell_scale
        self.shell_scale = shell_scale
        rule = solve_direction_rule(n_a, n_b)
        mu_d = np.array([rule["mu_a"], -rule["mu_a"], rule["mu_b"], -rule["mu_b"]])
        nz_d = np.array([n_a, -n_a, n_b, -n_b])
        W_d = np.array([rule["W_a"], rule["W_a"], rule["W_b"], rule["W_b"]]) / 2

        self.rule = rule
        self.v0 = rule["v0"]
        self.shell_p0 = p0_s
        self.shell_W = W_s
        self.shell = np.repeat(np.arange(n_shells), 4)
        self.p0 = np.repeat(p0_s, 4)
        self.mu = np.tile(mu_d, n_shells)
        self.pz = self.p0 * self.mu
        self.p_perp = self.p0 * np.sqrt(1 - self.mu ** 2)
        self.nz = np.tile(nz_d, n_shells)
        self.w = np.repeat(W_s, 4) * np.tile(W_d, n_shells)
        self.Q = self.p0.size
        self.name = f"{n_shells}sh x nz(+-{n_a},+-{n_b})" + (f" x T0={shell_scale:g}" if shell_scale != 1.0 else "")
        self.shifts_list = [(0, 0, int(n)) for n in self.nz]


def _report(st):
    print(f"== {st.name}: Q = {st.Q}, v0 = {st.v0:.6f} (Alessandro: 1/sqrt(41) = {1 / math.sqrt(41):.6f})")
    r = st.rule
    print(f"   mu_a = {r['mu_a']:.6f} (hop {r['n_a']}),  mu_b = {r['mu_b']:.6f} (hop {r['n_b']}),"
          f"  W_a = {r['W_a']:.6f},  W_b = {r['W_b']:.6f}")
    print("   direction moments <mu^k> vs 1/(k+1):")
    W_dir = np.array([r["W_a"], r["W_b"]])
    mu_dir = np.array([r["mu_a"], r["mu_b"]])
    for k in (0, 2, 4, 6, 8):
        val = (W_dir * mu_dir ** k).sum()
        print(f"     k={k}: {val:.6f}  exact {1 / (k + 1):.6f}  rel err {(val - 1 / (k + 1)) * (k + 1):+.2e}")
    print("   energy moments sum w p0^m vs (m+1)! T0^m:")
    for m in range(10):
        val = (st.w * st.p0 ** m).sum()
        ex = math.factorial(m + 1) * st.shell_scale ** m
        print(f"     m={m}: {val:.6e}  exact {ex:.6e}  rel err {val / ex - 1:+.2e}")


if __name__ == "__main__":
    st = Stencil16()
    _report(st)
    print("\n   nodes:   p0        mu        pz       nz   w")
    for i in range(st.Q):
        print(f"   {i:2d}  {st.p0[i]:8.4f}  {st.mu[i]:+.4f}  {st.pz[i]:+8.4f}  {st.nz[i]:+d}   {st.w[i]:.6e}")
