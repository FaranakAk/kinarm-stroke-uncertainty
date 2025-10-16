
"""
Train an evidential (digamma) network on Kinarm 3-task feature vectors.


Outputs:
  ./results/model_uncertainty_digamma_res_vgr_oh_oha_(swap)_{SEED}_{KEEP}.pt
"""

import argparse, os, random
import numpy as np
import torch
import torch.optim as optim

from helpers import get_device
from Kinarm import KINARM_FEATS_3TASKS
from train0 import train_model
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss
from model import ResNetBaseline

def set_seed(seed: int):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

def build_loaders(root, batch_size, keep_perc, sort_idx_path, num_workers=4):
    # Load split datasets
    train_ds = KINARM_FEATS_3TASKS(root=root, load=True, noise_type="clean", num_class=2, train="train")
    val_ds   = KINARM_FEATS_3TASKS(root=root, load=True, noise_type="clean", num_class=2, train="val")

    # Labels: last column is stroke/control; keep header for later analysis if needed
    train_header = train_ds.train_labels
    val_header   = val_ds.val_labels
    train_ds.train_labels = train_ds.train_labels[:, -1]
    val_ds.val_labels     = val_ds.val_labels[:, -1]

    # Optional pruning (keep % least-uncertain from prior run) — matches your notebook/code
    if keep_perc < 1.0:
        assert os.path.exists(sort_idx_path), f"Missing {sort_idx_path}"
        sort_ind = np.load(sort_idx_path)
        k = int(keep_perc * len(train_ds))
        keep = np.sort(sort_ind[:k])
        train_ds.train_data = train_ds.train_data[keep]
        train_ds.train_labels = train_ds.train_labels[keep]

    # Reshape to (N, C=1, F) for ResNetBaseline
    train_ds.train_data = torch.reshape(train_ds.train_data, (train_ds.train_data.shape[0], 1, train_ds.train_data.shape[-1]))
    val_ds.val_data     = torch.reshape(val_ds.val_data,     (val_ds.val_data.shape[0],     1, val_ds.val_data.shape[-1]))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    return {"train": train_loader, "val": val_loader}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/Kinarm/", help="Root folder for Kinarm NPZ/arrays")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2842137)
    ap.add_argument("--keep-perc", type=float, default=1.0, help="Fraction of training data to keep (0<..<=1).")
    ap.add_argument("--sort-ind", default="sort_ind_July14.npy", help="Indices sorted by uncertainty (lowest→highest).")
    ap.add_argument("--loss", choices=["digamma","log","mse"], default="digamma")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs("./results", exist_ok=True)

    loaders = build_loaders(root=args.root, batch_size=args.batch_size,
                            keep_perc=args.keep_perc, sort_idx_path=args.sort_ind)

    device = get_device()
    model = ResNetBaseline(in_channels=1, mid_channels=8, num_pred_classes=2).to(device)

    if args.loss == "digamma":
        criterion = edl_digamma_loss
    elif args.loss == "log":
        criterion = edl_log_loss
    else:
        criterion = edl_mse_loss

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # train_model: your helper returning (model, metrics)
    model, metrics = train_model(
        model,
        loaders,
        num_classes=2,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        uncertainty=True,
    )

    # Save checkpoint with your original naming style
    keep_tag = str(args.keep_perc).rstrip("0").rstrip(".") if args.keep_perc != int(args.keep_perc) else str(int(args.keep_perc))
    save_path = f"./results/model_uncertainty_{args.loss}_res_vgr_oh_oha_(swap)_{args.seed}_{keep_tag}.pt"
    state = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, save_path)
    print(f"[saved] {save_path}")

if __name__ == "__main__":
    main()
