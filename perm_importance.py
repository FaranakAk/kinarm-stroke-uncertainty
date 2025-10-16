
"""
Permutation importance on test split: shuffle one feature at a time and measure ΔAUC.
"""

import argparse, copy, numpy as np, torch
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from helpers import get_device
from Kinarm import KINARM_FEATS_3TASKS
from model import ResNetBaseline
from losses import relu_evidence

def predict_probs(model, X, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X)):
            x = Variable(X[i:i+1]).float().to(device)
            out = model(x)
            evid = relu_evidence(out)
            alpha = evid + 1
            p = (alpha / torch.sum(alpha, dim=1, keepdim=True))[:,1]
            preds.append(p.item())
    return np.asarray(preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/Kinarm/")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["test","val"], default="test")
    args = ap.parse_args()

    # Data
    if args.split == "test":
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="test")
        X = ds.test_data; y = ds.test_labels[:, -1]
    else:
        ds = KINARM_FEATS_3TASKS(root=args.root, load=True, noise_type="clean", num_class=2, train="val")
        X = ds.val_data;  y = ds.val_labels[:, -1]

    X = torch.reshape(X, (X.shape[0], 1, X.shape[-1]))
    y_true = np.array(y).astype(int).reshape(-1,)

    device = get_device()
    model = ResNetBaseline(in_channels=1, mid_channels=8, num_pred_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Baseline AUC using predicted labels (to mirror your code). Consider probs for a stricter AUC.
    base_preds = predict_probs(model, X, device)
    base_auc = roc_auc_score(y_true, (base_preds >= 0.5).astype(int))

    deltas = []
    X_copy = X.clone()
    for j in range(X.shape[-1]):
        # permute feature j across samples
        rp = torch.randperm(X.shape[0])
        X[:, 0, j] = X_copy[rp, 0, j]
        pj = predict_probs(model, X, device)
        auc_j = roc_auc_score(y_true, (pj >= 0.5).astype(int))
        deltas.append(base_auc - auc_j)
        X[:, 0, j] = X_copy[:, 0, j]  # restore

    deltas = np.array(deltas)
    print("[Permutation importance] ΔAUC per feature (baseline - shuffled):")
    print(deltas.tolist())

if __name__ == "__main__":
    main()
