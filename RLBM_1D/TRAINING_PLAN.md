# Training plan: NeurDE on a 16-node relativistic stencil

*Written 2026-10-11. Follows Phase 0 (`moment_budget.py`, `results/phase0_findings.md`) and the true-MJ check.*

## Goal

A NeurDE-style network that replaces the expensive part of Alessandro's relativistic LBM collision step (power-iteration eigensolve plus equilibrium) on a 16-node velocity set. It must reproduce the 128-velocity shock tube at matched physical times.

## Fixed choices

| Item | Choice |
|---|---|
| Velocity set | 4 energy shells × directions nz = ±2, ±5. `Stencil16((2, 5), shell_scale=0.75)` in `stencil16.py`. |
| Energy grid | Reference temperature T0 = 0.75 (300 MeV if T = 1 is 400 MeV). Shells: p0 = 0.557, 1.929, 4.298, 8.215. |
| Directions | μ = ±0.346 (hop 2), ±0.866 (hop 5). v0 = √3/10 = 0.1732. Exact up to ⟨μ⁴⟩; ⟨μ⁶⟩ is 0.8% high (accepted). |
| Network input | The 1D moments `(N3, T00, T03, T33) / N0`. N0 is used only to scale. |
| Network output | f_eq at the 16 nodes (positive, NeurDE's `exp` form) and the z-velocity `vz` |
| Collision | Alessandro's Anderson–Witting step, `f ← f − (UL/τ)(f − f_eq)` |
| Stage 1 target | Positive entropic f_eq on the 16 nodes matching the 128 run's f_eq moments of degree ≤ 2 (6 moments) |
| Stage 1 loss | NeurDE's relative error on f against the 16-node target (main term), plus a degree-3 moment term against the 128 run, plus `vz` |
| Stage 2 loss | `n`, `Uz`, `T` of the 16-node rollout against the 128 run at the same physical time |
| Moments of degree ≥ 4 | Never used in a loss. The 128 run is 20–50% wrong there at cold sites (checked against true MJ). |

Why degree 3 can come from the 128 data: the 128 run's f_eq matches true Maxwell–Jüttner exactly (to 1e-14) for degrees 0–3.

## Files

Everything goes in `NeurDE/RLBM_1D/` (public fork, branch `rlbm-1d`), except step 0, which goes in `relativistic_hydrodynamics` (private fork, branch `rlbm_neurde`). NeurDE's helpers are reused from `../SOD_shock_tube/utilities` (`calculate_relative_error`, optimizer and scheduler setup).

| File | Contents |
|---|---|
| `stencil16.py` | exists |
| `moments1d.py` | Moment matrix (powers of p0, pz up to degree 5), entropic fit, conservation projection. Moved out of `moment_budget.py` so training and analysis share one implementation. |
| `rlbm1d_solver.py` | 16-node solver, torch, GPU (step 1) |
| `run_classical16.py` | V3: classical 16-node run against the 128 run |
| `make_dataset.py` | Stage 1 data from the 128 runs |
| `model.py` | NeurDE model with the extra `vz` head |
| `train_stage_1.py`, `train_stage_2.py`, `eval.py` | Training and evaluation |
| `rlbm1d_params.yml` | All settings |
| `jobs/*.sbatch` | Perlmutter batch scripts (GPU node, `-C gpu`, `-A <account>`) |

## Step 0: more training data (`relativistic_hydrodynamics`)

A single shock-tube run can't show whether the network generalizes.

1. Make `init_bamps` take parameters: the density and temperature on each side, and the step steepness. The current BAMPS values stay as the default.
2. Add matching options to `generate_neurde_data.py`.
3. Generate about 4 more runs on Perlmutter: different temperature ratios (e.g. hot/cold = 1/0.6, 1/0.7, 0.9/0.5) and density ratios.
   - Keep every temperature in [0.5, 1]. That's the range the T0 = 0.75 grid was checked on.
   - Use 500 steps each, saving every step. Each run is about 6.5 GB on `/pscratch`.
4. Split:
   - **Train:** the original run's steps 1–400, plus 3 of the new runs.
   - **Test in time:** the original run's steps 401–500.
   - **Test on unseen conditions:** the 4th new run, never trained on.

## Step 1: 16-node solver (`rlbm1d_solver.py`)

A torch module, float64, with arrays of shape `(16, NZ)`. NZ = 12800, Δx = 12.8/12800 as in Alessandro's code, periodic.

- **`moments(f)`:** `N0, N3, T00, T03, T33` (in 1D the transverse components are zero by construction).
- **`landau(...)`:** closed-form 2×2 solve in place of the power iteration:
  ```
  ε  = [(T00 − T33) + sqrt((T00 + T33)² − 4·T03²)] / 2
  vz = (T00 − ε) / T03                      (vz = 0 where T03 = 0)
  U  = γ(1, vz)
  n  = U0·N0 − Uz·N3
  T  = ε / (3n)
  ```
- **`equilibrium(n, U, T, λ_prev)`:** the classical equilibrium. A positive entropic fit matching the exact MJ moments of degree ≤ 2, which have a closed form: `N^α = n U^α` and `T^{αβ} = 4nT·U^α U^β − nT·η^{αβ}`. Newton solve, warm-started from the previous step.
- **`project_conservation(f_eq, U, N, T)`:** the exact linear correction that makes the collision conserve mass, energy and momentum:
  ```
  Σ (p·U) f_eq      = U_β N^β
  Σ (p·U) p0 f_eq   = (T^{αβ} U_β)_0
  Σ (p·U) pz f_eq   = (T^{αβ} U_β)_z
  ```
- **`tau(n, T)`:** Alessandro's formula (`collide.py`) with v0 = 0.1732.
- **`collide(f, f_eq, U, τ)`:** `UL = U0 − Uz·pz/p0`, then `f − (UL/τ)(f − f_eq)`.
- **`stream(f)`:** `torch.roll` of each population by its hop `nz`.
- **`init_from_128(f128)`:** builds the starting 16-node state. It uses the positive entropic fit to the degree-≤2 moments of the 128 run's **f** (not f_eq), so the starting state also carries the non-equilibrium part, as far as degree 2 can.

**Time matching.** One step lasts `Δt = Δx / v0`. 16-node step `s` therefore sits at 128-velocity step `s · (1/√41) / v0 = 0.9017·s`. The 128 fields are interpolated linearly in time to each 16-node step.

## Step 2: V3 go/no-go, classical 16-node solver vs the 128 run (`run_classical16.py`)

Run the step-1 solver with **no network** from `init_from_128(f at step 1)` to the end of the 128 run (about 554 sixteen-node steps), for the original run and the held-out run. Report:

- relative L2 and maximum errors of `n`, `Uz`, `T`, `P = nT` against the 128 run at matched times
- the positions of the shocks and the rarefaction head
- totals of mass `Σ N0`, energy `Σ T00` and momentum `Σ T0z` over the grid: conserved to about 1e-12
- the fraction of steps where the Newton solve doesn't converge (expected 0, since degree ≤ 2 was always feasible in Phase 0)

**Gate:** agree the threshold with the supervisor (suggested: under 2–3% relative L2 in `n`, `T`, `P`, with no visible shift in the shock position). If it fails, the 16-node set itself can't carry this shock tube. Then reconsider 24 nodes before training anything.

These errors are also the **baseline** the trained network must match or beat.

## Step 3: Stage 1 data (`make_dataset.py`)

For each training run and step, pick sites:
- all non-equilibrium sites (`Σ|f − feq| / Σ|feq| > 1e-6`)
- about 1,000 random near-equilibrium sites, so that uniform regions don't dominate

For each site, store:

| Key | Contents |
|---|---|
| `x` | the input `(N3, T00, T03, T33) / N0`, computed from the 128 run's **f** (the same input the solver sees) |
| `N0` | for rescaling the output |
| `vz` | the 128 run's `Uz/U0`, plus `n` and `T` |
| `M_ref` | the 10 moments of degree ≤ 3 of the 128 run's f_eq, in the 16-node velocity set's units |
| `Feq16` | the positive entropic fit on the 16 nodes matching `M_ref`'s degree-≤2 part, plus its 6 multipliers `λ` |
| `noneq`, `T_bin` | flags for reporting by group |

**Checks before training:**
- `Feq16` matches the degree-≤2 moments to 1e-10 at every site
- all `Feq16 > 0`
- the degree-3 mismatch reproduces Phase 0 (about 2% at the 95th percentile)

## Step 4: model (`model.py`)

- `α = DenseNet([4, 64, 64, 64, 64])` on `x`. `φ = DenseNet([2, 64, 64, 64, 64])` on the node coordinates `(p0, μ)`. These are NeurDE's sizes.
- **f_eq:** `N0 · exp(α(x) · φ(p_i))`, which is positive and scales correctly with density.
- **vz head:** `vz = tanh(Linear(64→1)(α features))`, which keeps |vz| < 1. Then `U = γ(1, vz)`.

**Baseline to report:** `vz` from the closed-form `landau()` formula. In 1D, U can be computed exactly from the moments, so the head should be close to that. If it isn't, use the formula and drop the head.

## Step 5: Stage 1 training (`train_stage_1.py`)

```
L = ‖f − Feq16‖ / ‖Feq16‖                           (main term: NeurDE's relative error on f)
  + 0.1 · ‖A_3 f − M_ref,3‖ / ‖M_ref,3‖             (added term: degree-3 moments against the 128 run)
  + λ_U · mean (vz − vz_ref)²                        (λ_U = 1 to start)
```

The target `Feq16` already matches degrees 0–2 exactly, so the f term carries those.

`A_k` is the fixed matrix of degree-k powers on the 16 nodes.

The degree-3 term is what Phase 0 asked for: it trains the high-energy shells that a plain f-space loss barely sees. Where a positive solution with a better degree-3 match exists (about two thirds of sites), the network can beat `Feq16`.

Optimizer and schedule follow NeurDE's Stage 1 settings (AdamW, lr 1e-4, cosine annealing with warm restarts, gradient clipping at 1.0); adjust after the first run.

**V4 evaluation:** on the held-out steps and the held-out run, report:
- the moment error per degree, against `M_ref` and against `Feq16`
- the `vz` error, against the reference and against the closed-form baseline
- everything split into near-equilibrium/non-equilibrium and cold/hot sites

## Step 6: Stage 2 training (`train_stage_2.py`)

Windows of 25 sixteen-node steps, detaching between steps (as in NeurDE). Each window starts from `init_from_128` at a random training time. Each step:

1. `moments(f)`
2. network → `f_eq`, `vz`
3. `n = U·N`, `ε = U·T·U`, `T = ε/(3n)`, then `τ`
4. `project_conservation(f_eq)`
5. `collide`
6. `stream`

**Loss:** relative L2 of `n`, `Uz`, `T` against the 128 run, interpolated to the same time, summed over the window.

Start from the best Stage 1 model. Use NeurDE's Stage 2 settings (OneCycleLR, about 1,000 epochs) with a lower learning rate.

**V5 evaluation (`eval.py`):**
- full rollouts of the original run and the held-out run
- errors in `n`, `Uz`, `T`, `P` against the 128 run, compared with the V3 classical baseline
- conservation totals over time
- plots of the four fields at several times

**Success:** the network rollout is at least as accurate as V3 and conserves mass, energy and momentum to machine precision.

## Status (2026-10-13)

- The reduced velocity sets live in `relativistic_hydrodynamics/reduced_stencil.py`; `rlbm1d.py` uses them, so there is one source.
- V3 was run with Alessandro's code (`RLBM_STENCIL=16/24 python main.py`, `compare_runs.py`): n, ε, P within 1.4–2.1% (16 nodes) and 0.8–0.9% (24 nodes) relative L2 of the 128 run; `Uz` is the weakest field. Accepted as passing; training uses 16 nodes.
- Step 0 (extra runs) is postponed: training starts on the single run, split in time.
- `test_solver1d.py`: the torch 1D solver reproduces Alessandro's 16- and 24-node runs to 1e-15.

## Order and gates

1. **Step 0** (data runs) and **step 1** (solver) can go in parallel.
2. **Step 2 (V3) is the gate.** Don't train until it passes.
3. Steps 3 → 4 → 5, then check V4.
4. Step 6, then check V5.

## Open points (defaults chosen; revisit if needed)

- Thresholds for V3 and V5 (suggested 2–3% relative L2): agree with the supervisor.
- The extra runs' initial conditions (step 0).
- The weights in the Stage 1 loss (equal per degree, 0.1 for the f term).
- Whether to keep the `vz` head or use the closed-form U (decided by the V4 comparison).
- **Known limits:**
  - The model is strictly 1D, so the 3–6% transverse flow at the fronts in the 128 run is ignored.
  - The 16-node set is exact only to ⟨μ⁴⟩, so it misses the review's full order-3 condition by 0.8% in ⟨μ⁶⟩.
