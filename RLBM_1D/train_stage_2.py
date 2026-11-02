"""
Stage 2: train NeurDE1D inside the 16-node solver against the 128-velocity run
(TRAINING_PLAN.md, step 6).

Each window starts from a stored 16-node state (make_dataset.py, stage2.h5) and runs
`rollout` steps. One step:
  moments -> U, n, T (exact Landau frame of the state, as in NeurDE and the classical
  solver) -> network -> f_eq = N0 feq_hat -> conservation projection -> tau ->
  Anderson-Witting collision -> streaming
If the Stage 1 model takes the 8 x/y moment features (train_stage_1.py --xy-moments),
the frozen MomentMLP (train_moment_mlp.py) estimates them from the 4 t/z features; it
is stored in every Stage 2 checkpoint.
The loss compares n, Uz, T of the new state (exact Landau frame of its moments) with
the 128 run at the same physical time (linear interpolation between stored steps), on
the cells the 16-node set can reach (rlbm1d.reach_mask): beyond its fastest direction
no equilibrium can change anything, so those cells are left out.
Gradients reach the network through the collision of the same step; the state is
detached every `detach_every` steps (NeurDE: every step).

As in NeurDE, training runs every epoch (no early stopping) and keeps the `keep_top`
checkpoints with the lowest validation loss; best.pt is the lowest of them. The
Stage 1 model (epoch 0) competes too, so best.pt is never worse than it. The
classical 16-node solver's loss on the validation windows is logged as the baseline.

Usage: python train_stage_2.py --init checkpoints/stage1/best.pt [--params params.yml]
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import yaml

import rlbm1d as R
from model import NeurDE1D
from rollout import Stage2Data, load_moment_mlp, step_classical, step_network
from utilities import dispatch_optimizer, get_scheduler, set_seed  # NeurDE/SOD_shock_tube


def rollout_loss(step, data, k, st, rollout, weights, detach_every):
    """Mean loss over one window; `step` maps a state to the next one."""
    f = data.f_init[k]
    i0 = data.init_steps[k]
    mu_max = st.mu.abs().max().item()
    total = 0.0
    for s in range(1, rollout + 1):
        f = step(f)
        _, Uz, n, T = R.landau(*R.moments(f, st))
        t = i0 + s * data.ratio
        m = R.reach_mask(n.numel(), t * R.DT_128, mu_max).to(n.device)
        total = total + (weights["n"] * R.relL2(n[m], data.target("n", t)[m])
                         + weights["Uz"] * R.relL2(Uz[m], data.target("Uz", t)[m])
                         + weights["T"] * R.relL2(T[m], data.target("T", t)[m]))
        if s % detach_every == 0:
            f = f.detach()
    return total / rollout


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--init", required=True, help="Stage 1 checkpoint")
    ap.add_argument("--moment-mlp", default=os.path.join(R.HERE, "checkpoints", "moment_mlp", "best.pt"),
                    help="MomentMLP checkpoint, used when the Stage 1 model takes the x/y moment features")
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
    mlp, extra = None, {}
    if ckpt["cfg"]["model"].get("n_in", 4) == 12:
        mlp_state = ckpt.get("moment_mlp") or torch.load(args.moment_mlp, map_location=args.device)
        mlp = load_moment_mlp(mlp_state, args.device, dtype)
        extra = {"moment_mlp": mlp_state}  # stored with every checkpoint, so eval needs no path
        print(f"x/y moment features from the frozen MomentMLP (epoch {mlp_state.get('epoch')})")
    net_step = lambda f: step_network(model, f, st, mlp)
    optimizer = dispatch_optimizer(model, lr=c2["lr"], optimizer_type="AdamW")
    scheduler = get_scheduler(optimizer, c2["scheduler"], epochs * len(train_w), c2.get("scheduler_config", {}))

    model_dir = os.path.join(R.HERE, c2["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
    log = csv.writer(open(os.path.join(model_dir, "log.csv"), "w", newline=""))
    log.writerow(["epoch", "train_loss", "val_loss", "classical_val_loss", "lr", "seconds", "skipped_windows"])

    def val_loss(step):
        with torch.no_grad():
            return float(np.mean([rollout_loss(step, data, k, st, rollout, c2["loss_weights"], 1).item()
                                  for k in val_w])) if val_w else math.nan

    keep_top = c2.get("keep_top", 3)
    top = []  # (score, epoch, path), lowest score first

    def keep_if_top(score, epoch):
        """NeurDE's top-k checkpoints by validation loss; best.pt is the lowest."""
        if not math.isfinite(score) or (len(top) == keep_top and score >= top[-1][0]):
            return
        state = {"model": model.state_dict(), "cfg": ckpt["cfg"], "epoch": epoch, "val": score, **extra}
        path = os.path.join(model_dir, f"top_epoch{epoch:04d}.pt")
        torch.save(state, path)
        top.append((score, epoch, path))
        top.sort()
        while len(top) > keep_top:
            os.remove(top.pop()[2])
        if top[0][1] == epoch:
            torch.save(state, os.path.join(model_dir, "best.pt"))

    # Epoch 0: the classical solver (baseline) and the Stage 1 model before any Stage 2
    # training. The Stage 1 model is a checkpoint candidate like any later epoch.
    classical_val = val_loss(lambda f: step_classical(f, st))
    model.eval()
    val0 = val_loss(net_step)
    print(f"classical 16-node solver: val {classical_val:.3e}  (baseline)", flush=True)
    print(f"epoch    0  val {val0:.3e}  (Stage 1 model, before Stage 2)", flush=True)
    log.writerow([0, math.nan, val0, classical_val, optimizer.param_groups[0]["lr"], 0.0, 0])
    keep_if_top(val0 if val_w else math.inf, 0)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run, skipped = 0.0, 0
        for k in np.random.permutation(train_w):
            l = rollout_loss(net_step, data, k, st, rollout, c2["loss_weights"], c2["detach_every"])
            optimizer.zero_grad()
            if not torch.isfinite(l):  # a NaN/inf update would turn every weight into NaN
                skipped += 1
                scheduler.step()
                continue
            l.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad()
                scheduler.step()
                continue
            optimizer.step()
            scheduler.step()
            run += l.item()
        model.eval()
        val = val_loss(net_step)
        train = run / max(len(train_w) - skipped, 1)
        log.writerow([epoch, train, val, classical_val, optimizer.param_groups[0]["lr"], time.time() - t0, skipped])
        print(f"epoch {epoch:4d}  train {train:.3e}  val {val:.3e}  skipped {skipped}/{len(train_w)}"
              f"  lr {optimizer.param_groups[0]['lr']:.1e}  ({time.time() - t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "cfg": ckpt["cfg"], "epoch": epoch, "val": val, **extra},
                   os.path.join(model_dir, "last.pt"))
        # ranked by validation loss (training loss if there is no validation)
        keep_if_top(val if val_w else train, epoch)
    print(f"top {len(top)} by validation loss: "
          + ", ".join(f"epoch {e} ({s:.3e})" for s, e, _ in top)
          + f"; classical baseline {classical_val:.3e}; checkpoints in {model_dir}")


if __name__ == "__main__":
    main()
