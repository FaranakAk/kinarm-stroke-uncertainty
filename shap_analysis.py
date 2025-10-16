
"""
SHAP analysis for the evidential model on Kinarm features.
- Uses shap.DeepExplainer with a small background sample from the TRAIN split
- Saves: results/shap/summary_bar.png, summary_beeswarm.png, and optional per-sample waterfall plots
"""

import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

import shap
import pandas as pd

from helpers import get_device
from Kinarm import KINARM_FEATS_3TASKS
from model import ResNetBaseline

def to_channels_last_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Your ResNetBaseline expects input as (N, C=1, F). We stick to that.
    """
    if x.ndim == 2:
        x = torch.unsqueeze(x, 1)  # (N, 1, F)
    return x

def build_model(device):
    model = ResNetBaseline(in_channels=1, mid_channels=8, num_pred_classes=2).to(device)
    model.eval()
    return model

def load_split(root, split):
    ds = KINARM_FEATS_3TASKS(root=root, load=True, noise_type="clean", num_class=2, train=split)
    if split == "train":
        X = ds.train_data; y = ds.train_labels[:, -1]
    elif split == "val":
        X = ds.val_data;   y = ds.val_labels[:, -1]
    else:
        X = ds.test_data;  y = ds.test_labels[:, -1]
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/Kinarm/", help="Kinarm data root")
    ap.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    ap.add_argument("--split", choices=["test","val","train"], default="test", help="Split to explain")
    ap.add_argument("--background_size", type=int, default=200, help="DeepExplainer background sample size")
    ap.add_argument("--top_local", type=int, default=0, help="If >0, save waterfall plots for the first N samples")
    args = ap.parse_args()

    os.makedirs("results/shap", exist_ok=True)

    # Load model
    device = get_device()
    model = build_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load splits
    X_train, y_train = load_split(args.root, "train")
    X_exp,   y_exp   = load_split(args.root, args.split)

    # Torch tensors on CPU for DeepExplainer
    # (DeepExplainer works best when model & tensors are on same device; we keep CPU to be safe)
    model.cpu()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_exp_t   = torch.tensor(X_exp,   dtype=torch.float32)

    X_train_t = to_channels_last_1d(X_train_t)  # (N, 1, F)
    X_exp_t   = to_channels_last_1d(X_exp_t)    # (M, 1, F)

    # Build feature names like in your main.py ("feat0", "feat1", ...)
    F = X_exp_t.shape[-1]
    cols = [f"feat{j}" for j in range(F)]

    # Background for DeepExplainer
    bg_size = min(args.background_size, X_train_t.shape[0])
    if bg_size < 2:
        raise ValueError("Not enough training samples for DeepExplainer background.")
    bg_idx = np.random.choice(X_train_t.shape[0], size=bg_size, replace=False)
    background = X_train_t[bg_idx]  # (bg, 1, F)

    # Wrap model into a function SHAP can call: expects input tensor -> logits
    # Our model already does that.
    explainer = shap.DeepExplainer(model, background)

    # Compute SHAP values on the chosen split
    # DeepExplainer returns list [class0, class1] each (M, 1, F) shaped; we squeeze axes
    shap_values = explainer.shap_values(X_exp_t)
    # Squeeze to (M, F)
    if isinstance(shap_values, (list, tuple)) and len(shap_values) >= 2:
        sv_class0 = np.squeeze(shap_values[0], axis=1)
        sv_class1 = np.squeeze(shap_values[1], axis=1)
    else:
        # Fallback: single-output scenario
        sv_class0 = np.squeeze(shap_values, axis=1)
        sv_class1 = sv_class0

    # Build a DataFrame for pretty names (optional)
    X_exp_np = X_exp_t.squeeze(1).numpy()  # (M, F)
    test_df = pd.DataFrame(X_exp_np, columns=cols)

    # --- Global plots ---
    plt.close("all")
    shap.summary_plot(sv_class1, test_df.values, plot_type="bar", feature_names=test_df.columns, show=False)
    plt.tight_layout()
    plt.savefig("results/shap/summary_bar.png", dpi=180, bbox_inches="tight")

    plt.close("all")
    shap.summary_plot(sv_class1, test_df.values, feature_names=test_df.columns, show=False)
    plt.tight_layout()
    plt.savefig("results/shap/summary_beeswarm.png", dpi=180, bbox_inches="tight")

    # --- Local plots (optional) ---
    # Use expected_value for class 0 index (because SHAP returns per-class arrays); you used index 0 in your code.
    if args.top_local > 0:
        exp_val = explainer.expected_value
        # expected_value can be a list (per-class). Use class-0 for consistency with your code.
        if isinstance(exp_val, (list, tuple)):
            exp0 = exp_val[0]
        else:
            exp0 = exp_val

        for i in range(min(args.top_local, test_df.shape[0])):
            # Legacy waterfall (matches your notebook’s call)
            plt.close("all")
            shap.plots._waterfall.waterfall_legacy(exp0, sv_class0[i], feature_names=test_df.columns, show=False)
            plt.tight_layout()
            plt.savefig(f"results/shap/waterfall_sample{i}.png", dpi=180, bbox_inches="tight")

    print("[SHAP] Wrote:")
    print("  results/shap/summary_bar.png")
    print("  results/shap/summary_beeswarm.png")
    if args.top_local > 0:
        print(f"  results/shap/waterfall_sample0..{args.top_local-1}.png")

if __name__ == "__main__":
    main()
