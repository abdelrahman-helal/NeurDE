"""
Train MomentMLP: the 8 x/y moment features (N1, N2, T01, T02, T12, T13, T23, T11 - T22,
divided by N0) of the 128 run's f, from its 4 t/z features (N3, T00, T03, T33) / N0.

A reduced set has no x/y entries, so with train_stage_1.py --xy-moments the Stage 2
rollout feeds these estimates where Stage 1 saw the 128 run's values.
Loss: mean squared error in units of each target's standard deviation.
Every epoch logs, per output, the validation RMSE of the MLP and of predicting zero:
a ratio near 1 means the x/y entries are not predictable from the t/z features.

Usage: python train_moment_mlp.py [--params params.yml] [--device cuda] [--epochs N]
"""

import argparse
import csv
import os
import time

import h5py
import numpy as np
import torch
import yaml

import rlbm1d as R
from model import MomentMLP
from utilities import dispatch_optimizer, get_scheduler, set_seed  # NeurDE/SOD_shock_tube

NAMES = ("N1", "N2", "T01", "T02", "T12", "T13", "T23", "T11-T22")


@torch.no_grad()
def rmse(model, x, y, batch):
    pred = torch.cat([model(xb) for xb in torch.split(x, batch)])
    return ((pred - y) ** 2).mean(0).sqrt(), (y ** 2).mean(0).sqrt()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", default=os.path.join(R.HERE, "params.yml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-steps-max", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.params))
    cm = cfg["moment_mlp"]
    epochs = args.epochs or cm["epochs"]
    train_max = args.train_steps_max or cfg["train_steps_max"]
    dtype = getattr(torch, cfg["dtype"])
    set_seed(args.seed)

    t = lambda a: torch.tensor(a, device=args.device, dtype=dtype)
    with h5py.File(os.path.join(R.HERE, cfg["data_dir"], "stage1.h5"), "r") as h:
        if "moments_xy" not in h:
            raise SystemExit("stage1.h5 has no moments_xy: re-run make_dataset.py")
        mom, mxy, step = t(h["moments"][:]), t(h["moments_xy"][:]), h["step"][:]
    x, y = R.features(*mom.T), R.xy_features(mxy, mom[:, 0])
    tr, va = np.flatnonzero(step < train_max), np.flatnonzero(step >= train_max)
    print(f"moment MLP: {len(tr)} training samples, {len(va)} validation samples")

    model = MomentMLP(cm["hidden"], cm["layers"], cm["activation"]).to(args.device, dtype)
    model.set_normalization(x[tr], y[tr])
    optimizer = dispatch_optimizer(model, lr=cm["lr"], optimizer_type="AdamW")
    batches = int(np.ceil(len(tr) / cm["batch_size"]))
    sc = dict(cm.get("scheduler_config", {}))
    if "T_0_epochs" in sc:
        sc["T_0"] = sc.pop("T_0_epochs") * batches
    scheduler = get_scheduler(optimizer, cm["scheduler"], batches * epochs, sc)

    model_dir = os.path.join(R.HERE, cm["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
    log = csv.writer(open(os.path.join(model_dir, "log.csv"), "w", newline=""))
    log.writerow(["epoch", "train_loss", "val_loss"] + [f"val_ratio_{n}" for n in NAMES] + ["lr", "seconds"])
    best = np.inf
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for b in torch.split(torch.as_tensor(np.random.permutation(tr), device=args.device), cm["batch_size"]):
            l = (((model(x[b]) - y[b]) / model.y_std) ** 2).mean()
            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            run += l.item() * len(b)
        model.eval()
        err, zero = rmse(model, x[va], y[va], cm["batch_size"])
        val = float(((err / model.y_std) ** 2).mean())
        ratio = (err / zero.clamp(min=1e-300)).tolist()
        log.writerow([epoch, run / len(tr), val] + ratio + [optimizer.param_groups[0]["lr"], time.time() - t0])
        print(f"epoch {epoch:4d}  train {run / len(tr):.3e}  val {val:.3e}  RMSE / RMSE(zero): "
              + " ".join(f"{n} {r:.2f}" for n, r in zip(NAMES, ratio)) + f"  ({time.time() - t0:.0f}s)", flush=True)
        state = {"model": model.state_dict(), "cfg": cm, "epoch": epoch, "val": val, "ratio": ratio}
        torch.save(state, os.path.join(model_dir, "last.pt"))
        if val < best:
            best = val
            torch.save(state, os.path.join(model_dir, "best.pt"))

    err, zero = rmse(model, x[va], y[va], cm["batch_size"])
    print("\nlast epoch, validation    RMSE        RMSE(zero)   ratio")
    for n, e, z in zip(NAMES, err.tolist(), zero.tolist()):
        print(f"  {n:8s}            {e:.3e}   {z:.3e}    {e / max(z, 1e-300):.2f}")
    print(f"best validation loss {best:.3e}; checkpoints in {model_dir}")


if __name__ == "__main__":
    main()
