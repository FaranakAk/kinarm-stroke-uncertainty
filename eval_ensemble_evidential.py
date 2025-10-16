
"""
Ensemble evaluation for five evidential checkpoints (averaged logits).
"""

import argparse, os
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from helpers import get_device
from Kinarm import KINARM_FEATS_3TASKS
from model import ResNetBaseline
from losses import relu_evidence

SEEDS = [4779059, 9850562, 1998978, 2842137, 136247]  # your originals

def cm_calc(y_true, y_pred):
    cm = np.zeros((2,2))
    for i in range(len(y_true)):
        i1 = 0 if y_true[i]==1 else 1
        i2 = 0 if y_pred[i]==1 else 1
        cm[i1,i2] += 1
    return cm

def cm_metric(cm):
    acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
    sen = cm[0,0] / np.sum(cm[0,]) if np.sum(cm[0,]) else 0.0
    spe = cm[1,1] / np.sum(cm[1,]) if np.sum(cm[1,]) else 0.0
    return acc, sen, spe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/Kinarm/")
    ap.add_argument("--keep-perc", type=float, default=1.0, help="Which keep% your checkpoints used.")
    ap.add_argument("--loss", choices=["digamma"], default="digamma")
    ap.add_argument("--split", choices=["test","val","train"], default="test")
    args = ap.parse_args()

    # Dataset
    if args.split == "test":
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="test")
        X = ds.test_data; y = ds.test_labels[:, -1]
    elif args.split == "val":
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="val")
        X = ds.val_data;  y = ds.val_labels[:, -1]
    else:
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="train")
        X = ds.train_data; y = ds.train_labels[:, -1]

    # Reshape
    X = torch.reshape(X, (X.shape[0], 1, X.shape[-1]))
    y_true = np.array(y).astype(int).reshape(-1,)

    device = get_device()
    models = []
    for s in SEEDS:
        m = ResNetBaseline(in_channels=1, mid_channels=8, num_pred_classes=2).to(device)
        keep_tag = str(args.keep_perc).rstrip("0").rstrip(".") if args.keep_perc != int(args.keep_perc) else str(int(args.keep_perc))
        ckpt_path = f"./results/model_uncertainty_{args.loss}_res_vgr_oh_oha_{s}_2_{keep_tag}.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"./results/model_uncertainty_{args.loss}_res_vgr_oh_oha_(swap)_{s}_{keep_tag}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)

    preds = []; probs = []
    with torch.no_grad():
        for i in range(len(X)):
            x = Variable(X[i:i+1]).float().to(device)
            logits = None
            for m in models:
                out = m(x)
                logits = out if logits is None else (logits + out)
            logits = logits / len(models)
            evid = relu_evidence(logits)
            alpha = evid + 1
            _, pcls = torch.max(logits, 1)
            p = (alpha / torch.sum(alpha, dim=1, keepdim=True))[:,1]
            preds.append(pcls.item()); probs.append(p.item())

    preds = np.asarray(preds); probs = np.asarray(probs)
    cm = cm_calc(y_true, preds)
    acc, sen, spe = cm_metric(cm)
    auc = roc_auc_score(y_true, preds)

    print(f"ACC={acc:.3f}  SEN={sen:.3f}  SPE={spe:.3f}  AUC={auc:.3f}")

if __name__ == "__main__":
    main()
