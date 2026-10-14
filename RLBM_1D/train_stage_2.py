"""
Stage 2: train NeurDE1D inside the 16-node solver against the 128-velocity run
(TRAINING_PLAN.md, step 6).

Each window starts from a stored 16-node state (make_dataset.py, stage2.h5) and runs
`rollout` steps. One step:
  moments -> network (feq_hat, vz) -> f_eq = N0 feq_hat, U = gamma (1, vz) -> conservation
  projection -> n, T, tau -> Anderson-Witting collision -> streaming
The loss compares n, Uz, T of the new state (exact Landau frame of its moments) with
the 128 run at the same physical time (linear interpolation between stored steps).
Gradients reach the network through the collision of the same step; the state is
detached every `detach_every` steps (NeurDE: every step).

Usage: python train_stage_2.py --init checkpoints/stage1/best.pt [--params params.yml]
"""

import argparse
import csv
import math
import os
import time

import h5py
import numpy as np
import torch
import yaml

import rlbm1d as R
from model import NeurDE1D
from utilities import dispatch_optimizer, get_scheduler, set_seed  # NeurDE/SOD_shock_tube


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


def step_network(model, f, st):
    """One 16-node step with the network equilibrium. Returns the new state."""
    mom = R.moments(f, st)
    feq_hat, vz = model(R.features(*mom), st.basis)  # feq_hat = f_eq / N0
    U0 = 1 / torch.sqrt(1 - vz ** 2)
    Uz = U0 * vz
    feq = R.project_conservation(mom[0][None] * feq_hat.T, U0, Uz, *mom, st)
    n, _, T = R.n_eps_T(U0, Uz, *mom)
    return R.stream(R.collide(f, feq, U0, Uz, R.tau(n, T, st.v0), st), st)


def rollout_loss(model, data, k, st, rollout, weights, detach_every):
    f = data.f_init[k]
    i0 = data.init_steps[k]
    total = 0.0
    for s in range(1, rollout + 1):
        f = step_network(model, f, st)
        _, Uz, n, T = R.landau(*R.moments(f, st))
        t = i0 + s * data.ratio
        total = total + (weights["n"] * R.relL2(n, data.target("n", t))
                         + weights["Uz"] * R.relL2(Uz, data.target("Uz", t))
                         + weights["T"] * R.relL2(T, data.target("T", t)))
        if s % detach_every == 0:
            f = f.detach()
    return total / rollout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--init", required=True, help="Stage 1 checkpoint")
    ap.add_argument("--params", default=os.path.join(R.HERE, "params.yml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rollout", type=int, default=None)
    ap.add_argument("--train-steps-max", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.params))
    c2 = cfg["stage2"]
    epochs = args.epochs or c2["epochs"]
    rollout = args.rollout or c2["rollout"]
    train_max = args.train_steps_max or cfg["train_steps_max"]
    dtype = getattr(torch, cfg["dtype"])
    set_seed(args.seed)

    st = R.Stencil1D(cfg["stencil"]["hops"], cfg["stencil"]["T0"], device=args.device, dtype=dtype)
    data = Stage2Data(os.path.join(R.HERE, cfg["data_dir"], "stage2.h5"), args.device, dtype)
    train_w = data.windows(rollout, 0, train_max - 1)
    val_w = data.windows(rollout, train_max, data.n_steps - 1)
    print(f"{st.name}: {len(train_w)} training windows, {len(val_w)} validation windows, "
          f"{rollout} steps each ({rollout * data.ratio:.1f} steps of the 128 run)")
    if not train_w:
        raise SystemExit("no training windows: check rollout, train_steps_max and make_dataset --init-every")

    ckpt = torch.load(args.init, map_location=args.device)
    model = NeurDE1D(**ckpt["cfg"]["model"]).to(args.device, dtype)
    model.load_state_dict(ckpt["model"])
    optimizer = dispatch_optimizer(model, lr=c2["lr"], optimizer_type="AdamW")
    scheduler = get_scheduler(optimizer, c2["scheduler"], epochs * len(train_w), c2.get("scheduler_config", {}))

    model_dir = os.path.join(R.HERE, c2["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
    log = csv.writer(open(os.path.join(model_dir, "log.csv"), "w", newline=""))
    log.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])
    best = math.inf
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for k in np.random.permutation(train_w):
            l = rollout_loss(model, data, k, st, rollout, c2["loss_weights"], c2["detach_every"])
            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            run += l.item()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([rollout_loss(model, data, k, st, rollout, c2["loss_weights"], 1).item()
                                 for k in val_w])) if val_w else math.nan
        train = run / len(train_w)
        log.writerow([epoch, train, val, optimizer.param_groups[0]["lr"], time.time() - t0])
        print(f"epoch {epoch:4d}  train {train:.3e}  val {val:.3e}  ({time.time() - t0:.0f}s)", flush=True)
        state = {"model": model.state_dict(), "cfg": ckpt["cfg"], "epoch": epoch, "val": val}
        torch.save(state, os.path.join(model_dir, "last.pt"))
        if val < best or (math.isnan(val) and epoch == epochs):
            best = val
            torch.save(state, os.path.join(model_dir, "best.pt"))
    print(f"best validation loss {best:.3e}; checkpoints in {model_dir}")


if __name__ == "__main__":
    main()
