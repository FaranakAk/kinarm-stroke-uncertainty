
"""
Evaluate a trained evidential model:
- Loads Kinarm test split
- Computes ACC/SEN/SPE/AUC
- Optional uncertainty filtering via --uncert-thr
"""

import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from helpers import get_device
from Kinarm import KINARM_FEATS_3TASKS
from model import ResNetBaseline
from losses import relu_evidence

def cmsa_binary(y):  # if labels are already 0/1, this is identity; kept for clarity
    y = np.array(y).astype(int)
    return y.reshape(-1,)

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
    ap.add_argument("--checkpoint", required=True, help="Path to .pt")
    ap.add_argument("--split", choices=["test","val","train"], default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--uncert-thr", type=float, default=None, help="Keep samples with uncertainty <= thr.")
    args = ap.parse_args()

    # Load dataset
    if args.split == "test":
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="test")
        X = ds.test_data; y = ds.test_labels[:, -1]
    elif args.split == "val":
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="val")
        X = ds.val_data;  y = ds.val_labels[:, -1]
    else:
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="train")
        X = ds.train_data; y = ds.train_labels[:, -1]

    # Reshape to (N, 1, F)
    X = torch.reshape(X, (X.shape[0], 1, X.shape[-1]))
    y_true = cmsa_binary(y)

    device = get_device()
    model = ResNetBaseline(in_channels=1, mid_channels=8, num_pred_classes=2).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    probs = []
    uncerts = []
    with torch.no_grad():
        for i in range(len(X)):
            x = Variable(X[i:i+1]).float().to(device)
            out = model(x)
            # Evidential:
            evid = relu_evidence(out)
            alpha = evid + 1
            uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
            _, pcls = torch.max(out, 1)
            p = (alpha / torch.sum(alpha, dim=1, keepdim=True))[:,1]  # prob of class=1
            preds.append(pcls.item())
            probs.append(p.item())
            uncerts.append(uncertainty.item())

    preds = np.asarray(preds); probs = np.asarray(probs); uncerts = np.asarray(uncerts)

    # Optional uncertainty filtering
    keep = np.arange(len(preds))
    if args.uncert_thr is not None:
        keep = np.where(uncerts <= args.uncert_thr)[0]
        if keep.size == 0:
            print("[warn] no samples <= uncertainty threshold; using all samples.")
            keep = np.arange(len(preds))

    yk = y_true[keep]; pk = preds[keep]; sk = probs[keep]
    cm = cm_calc(yk, pk)
    acc, sen, spe = cm_metric(cm)
    auc = roc_auc_score(yk, pk)  # AUC over predicted labels (as in your script); change to sk for score-based AUC

    print(f"Samples kept: {len(keep)}/{len(y_true)}")
    print(f"ACC={acc:.3f}  SEN={sen:.3f}  SPE={spe:.3f}  AUC={auc:.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.csv", "w") as f:
        f.write("checkpoint,split,n_kept,acc,sen,spe,auc,uncert_thr\n")
        f.write(f"{os.path.basename(args.checkpoint)},{args.split},{len(keep)},{acc:.6f},{sen:.6f},{spe:.6f},{auc:.6f},{args.uncert_thr}\n")
    print("[write] results/metrics.csv")

if __name__ == "__main__":
    main()
