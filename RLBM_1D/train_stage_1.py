"""
Stage 1: fit NeurDE1D to the equilibrium of the 128-velocity run (TRAINING_PLAN.md, step 5).

Loss per batch, in units of N0 (feq_hat = f_eq / N0, the network output):
  w_f ||feq_hat - Feq16/N0|| / ||Feq16/N0||   main term: NeurDE's relative error on f against
                                              the positive 16-node target (degree <= 2 exact)
  + w_3 ||A_3 feq_hat - Mref_3/N0|| / ||.||   added term: degree-3 moments against the 128 run
  + w_vz mean (vz - vz_ref)^2
Training: h5 steps < train_steps_max; validation: the remaining steps.
Validation also reports the per-site moment error |M - Mref| / sum|feq| p0^k (95th
percentile, as in Phase 0) and the closed-form vz baseline from the input moments.

Usage: python train_stage_1.py [--params params.yml] [--device cuda] [--epochs N]
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
from model import NeurDE1D
from utilities import calculate_relative_error, dispatch_optimizer, get_scheduler, set_seed  # NeurDE/SOD_shock_tube


def load_stage1(path, device, dtype):
    t = lambda a: torch.tensor(a, device=device, dtype=dtype)
    with h5py.File(path, "r") as h:
        mom = h["moments"][:]
        d = dict(mom=t(mom), Mref=t(h["Mref"][:]), lead=t(h["lead"][:]), Feq16=t(h["Feq16"][:]),
                 vz=t(h["vz"][:]), step=h["step"][:], noneq=h["noneq"][:].astype(bool), T=h["T"][:])
    N0 = d["mom"][:, 0]
    d["x"] = R.features(*d["mom"].T)
    # Everything the loss compares is divided by N0, like the network output feq_hat
    d["feq_hat_target"] = d["Feq16"] / N0[:, None]
    d["M_hat_target"] = d["Mref"] / N0[:, None]
    d["lead_hat"] = d["lead"] / N0[:, None]
    return d


class Stage1Loss:
    def __init__(self, st, weights):
        kb3 = R.monomial_list(3)
        self.A = R.monomials(st.p0, st.pz, kb3)                      # (10, Q)
        self.deg = torch.tensor([k for k, _ in kb3], device=st.p0.device)
        self.w = weights

    def __call__(self, feq_hat, vz, d, idx):
        d3 = self.deg == 3
        parts = {"f": calculate_relative_error(feq_hat, d["feq_hat_target"][idx]),
                 "moment3": calculate_relative_error(feq_hat @ self.A[d3].T, d["M_hat_target"][idx][:, d3]),
                 "vz": ((vz - d["vz"][idx]) ** 2).mean()}
        total = self.w["f"] * parts["f"] + self.w["moment3"] * parts["moment3"] + self.w["vz"] * parts["vz"]
        return total, parts

    def site_errors(self, feq_hat, d, idx):
        """Phase 0 metric: worst monomial of each degree, |M - Mref| / sum|feq| p0^k."""
        rel = (feq_hat @ self.A.T - d["M_hat_target"][idx]).abs() / d["lead_hat"][idx][:, self.deg]
        return torch.stack([rel[:, self.deg == k].amax(1) for k in range(4)], 1)


@torch.no_grad()
def evaluate(model, loss, st, d, idx, batch):
    tot, parts, errs, vzs = 0.0, {}, [], []
    for b in torch.split(torch.as_tensor(idx, device=st.p0.device), batch):
        feq_hat, vz = model(d["x"][b], st.basis)
        l, p = loss(feq_hat, vz, d, b)
        tot += l.item() * len(b)
        for k, v in p.items():
            parts[k] = parts.get(k, 0.0) + v.item() * len(b)
        errs.append(loss.site_errors(feq_hat, d, b).cpu().numpy())
        vzs.append((vz - d["vz"][b]).abs().cpu().numpy())
    n = len(idx)
    errs, vzs = np.concatenate(errs), np.concatenate(vzs)
    out = {"loss": tot / n, **{k: v / n for k, v in parts.items()}}
    noneq = d["noneq"][idx]
    for k in range(4):
        out[f"p95_deg{k}"] = float(np.percentile(errs[:, k], 95))
        out[f"p95_deg{k}_noneq"] = float(np.percentile(errs[noneq, k], 95)) if noneq.any() else np.nan
    out["vz_mae"], out["vz_max"] = float(vzs.mean()), float(vzs.max())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", default=os.path.join(R.HERE, "params.yml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-steps-max", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.params))
    c1 = cfg["stage1"]
    epochs = args.epochs or c1["epochs"]
    train_max = args.train_steps_max or cfg["train_steps_max"]
    dtype = getattr(torch, cfg["dtype"])
    set_seed(args.seed)

    st = R.Stencil1D(cfg["stencil"]["hops"], cfg["stencil"]["T0"], device=args.device, dtype=dtype)
    data_dir = os.path.join(R.HERE, cfg["data_dir"])
    d = load_stage1(os.path.join(data_dir, "stage1.h5"), args.device, dtype)
    train_idx = np.flatnonzero(d["step"] < train_max)
    val_idx = np.flatnonzero(d["step"] >= train_max)
    print(f"{st.name}: {len(train_idx)} training samples, {len(val_idx)} validation samples")

    # Closed-form vz from the input moments: the baseline the vz head should approach
    U0, Uz, _, _ = R.landau(*d["mom"][val_idx].T)
    base = (Uz / U0 - d["vz"][val_idx]).abs()
    print(f"closed-form vz baseline on validation: mean |err| {base.mean():.2e}, max {base.max():.2e}")

    model = NeurDE1D(**cfg["model"]).to(args.device, dtype)
    model.set_normalization(d["x"][train_idx])
    loss = Stage1Loss(st, c1["loss_weights"])
    optimizer = dispatch_optimizer(model, lr=c1["lr"], optimizer_type="AdamW")
    batches = int(np.ceil(len(train_idx) / c1["batch_size"]))
    sc = dict(c1.get("scheduler_config", {}))
    if "T_0_epochs" in sc:
        sc["T_0"] = sc.pop("T_0_epochs") * batches
    scheduler = get_scheduler(optimizer, c1["scheduler"], batches * epochs, sc)

    model_dir = os.path.join(R.HERE, c1["model_dir"])
    os.makedirs(model_dir, exist_ok=True)
    log = open(os.path.join(model_dir, "log.csv"), "w", newline="")
    writer = None
    best = np.inf
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.as_tensor(np.random.permutation(train_idx), device=args.device)
        run = 0.0
        for b in torch.split(perm, c1["batch_size"]):
            feq_hat, vz = model(d["x"][b], st.basis)
            l, _ = loss(feq_hat, vz, d, b)
            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            run += l.item() * len(b)
        model.eval()
        val = evaluate(model, loss, st, d, val_idx, c1["batch_size"])
        row = {"epoch": epoch, "train_loss": run / len(train_idx), **{f"val_{k}": v for k, v in val.items()},
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        log.flush()
        print(f"epoch {epoch:4d}  train {row['train_loss']:.3e}  val {val['loss']:.3e}  "
              f"p95 deg0-3 {val['p95_deg0']:.1e} {val['p95_deg1']:.1e} {val['p95_deg2']:.1e} {val['p95_deg3']:.1e}  "
              f"vz mae {val['vz_mae']:.1e}  ({row['seconds']:.0f}s)", flush=True)
        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": val}
        torch.save(ckpt, os.path.join(model_dir, "last.pt"))
        if val["loss"] < best:
            best = val["loss"]
            torch.save(ckpt, os.path.join(model_dir, "best.pt"))
    print(f"best validation loss {best:.3e}; checkpoints in {model_dir}")


if __name__ == "__main__":
    main()
