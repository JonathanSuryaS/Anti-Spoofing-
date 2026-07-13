#!/usr/bin/env python3
"""
compare_checkpoints.py — baseline vs SPSC vs masked, cross-dataset
==================================================================

Each checkpoint is evaluated on THE DATA ITS PIPELINE PRODUCES:

    baseline / spsc  ->  data/processed_cropped   (unmasked crops)
    masked           ->  data/processed_masked    (background suppressed)

This matters. The masked model never saw a background during training; feeding
it unmasked images (or feeding the baseline masked ones) is a preprocessing
mismatch and the resulting numbers are meaningless. So the table compares
PIPELINES (preprocessing + model), not models on identical inputs. That is the
deployment-realistic comparison — in production the masked model would always
receive masked input — but say so in the writeup, because the differing test
inputs per row is the first thing a reviewer will query.

Writes into ./comparison_out/:
    comparison_results.md / .csv
    roc_<domain>.png      all models overlaid
    scores_<domain>.png   P(spoof) histograms per model

Label convention:  live = 0 (bonafide), spoof = 1 (attack = positive class)
Spoof score     :  softmax(logits)[:, 1]

Metrics (predict spoof if score >= tau):
    APCER = fraction of SPOOF samples with score <  tau   (attacks that passed)
    BPCER = fraction of LIVE  samples with score >= tau   (real users rejected)
    ACER  = (APCER + BPCER) / 2
    EER   = the point where APCER == BPCER

    BPCER @ APCER=1%  = the SECURITY OPERATING POINT. Fix the threshold so only
    1% of attacks get through, then ask how many real users that rejects. This
    is the number that matters for a real system, and — unlike APCER/BPCER@0.5 —
    it is not measured at a threshold both domains have long since abandoned.
    (OULU's optimal threshold is ~0.05, Kaggle's ~0.99. A fixed 0.5 cutoff sits
    in dead space for both, which is why those columns barely move even when the
    model changes a lot.)
"""

import os
import sys
import glob
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ CONFIG ------------------------------
CROPPED_ROOT = "data/processed_cropped"
MASKED_ROOT  = "data/processed_masked"

# name -> (checkpoint path, data root it must be evaluated on)
CHECKPOINTS = {
    "baseline": ("results/checkpoints/resnet50_baseline.pth", CROPPED_ROOT),
    "spsc":     ("results/checkpoints/resnet50_spsc.pth",     CROPPED_ROOT),
    "masked":   ("results/checkpoints/resnet50_masked.pth",   MASKED_ROOT),
}

TEST_DOMAINS = ["test_oulu", "test_kaggle"]
OUT_DIR      = "comparison_out"

TARGET_APCER = 0.01        # security operating point: let through 1% of attacks

IMG_SIZE     = 224
BATCH_SIZE   = 64
NUM_WORKERS  = 0           # keep 0 on Windows
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS      = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
# --------------------------------------------------------------------


class DomainDataset(Dataset):
    """Reads <domain_dir>/{live,spoof}/* -> (tensor, label). live=0, spoof=1."""

    def __init__(self, domain_dir, tf):
        self.samples = []
        for label, sub in [(0, "live"), (1, "spoof")]:
            d = os.path.join(domain_dir, sub)
            files = []
            for ext in IMG_EXTS:
                files += glob.glob(os.path.join(d, ext))
            self.samples += [(f, label) for f in sorted(files)]
        self.tf = tf
        if not self.samples:
            raise FileNotFoundError(f"No images under {domain_dir}/(live|spoof).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.tf(Image.open(path).convert("RGB")), label


eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model():
    m = resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    ckpt = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"  [warn] {len(missing)} missing / {len(unexpected)} unexpected keys")
    return model


@torch.no_grad()
def get_scores(model, loader):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        p = F.softmax(model(x.to(DEVICE)), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(scores), np.concatenate(labels)


def metrics_at(scores, labels, tau):
    spoof, live = labels == 1, labels == 0
    apcer = float(np.mean(scores[spoof] <  tau)) if spoof.any() else float("nan")
    bpcer = float(np.mean(scores[live]  >= tau)) if live.any()  else float("nan")
    return apcer, bpcer, (apcer + bpcer) / 2


def compute_eer(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def bpcer_at_apcer(scores, labels, target_apcer=TARGET_APCER):
    """
    Security operating point.

    Raise the threshold until at most `target_apcer` of attacks slip through,
    then report how many genuine users that setting rejects. Returns
    (bpcer, threshold, achieved_apcer) — achieved_apcer is reported because on
    small test sets the exact target may not be reachable and you should know
    what you actually got.
    """
    spoof_scores = np.sort(scores[labels == 1])
    if spoof_scores.size == 0:
        return float("nan"), float("nan"), float("nan")

    # APCER = P(spoof score < tau) grows monotonically with tau, so we want the
    # LARGEST tau whose APCER is still within budget. Scan sorted candidates and
    # stop at the first violation.
    cands = np.unique(scores)
    best_tau, best_apcer = None, None
    for tau in cands:
        apcer = float(np.mean(spoof_scores < tau))
        if apcer <= target_apcer:
            best_tau, best_apcer = float(tau), apcer
        else:
            break   # cands is sorted; APCER only grows from here

    if best_tau is None:
        return float("nan"), float("nan"), float("nan")

    bpcer = float(np.mean(scores[labels == 0] >= best_tau))
    return bpcer, best_tau, best_apcer


def evaluate(scores, labels):
    auc = roc_auc_score(labels, scores)
    eer, eer_thr = compute_eer(scores, labels)
    a5, b5, c5 = metrics_at(scores, labels, 0.5)
    ae, be, ce = metrics_at(scores, labels, eer_thr)
    bp_sec, tau_sec, apcer_sec = bpcer_at_apcer(scores, labels)
    return dict(auc=auc, eer=eer, eer_thr=eer_thr,
                apcer50=a5, bpcer50=b5, acer50=c5,
                apcer_eer=ae, bpcer_eer=be, acer_eer=ce,
                bpcer_sec=bp_sec, tau_sec=tau_sec, apcer_sec=apcer_sec)


def plot_roc(domain, per_model, out_path):
    plt.figure(figsize=(5.5, 5.5))
    for name, d in per_model.items():
        fpr, tpr, _ = roc_curve(d["labels"], d["scores"], pos_label=1)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={d['m']['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("FPR (BPCER)")
    plt.ylabel("TPR (1 - APCER)")
    plt.title(f"ROC — {domain}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scores(domain, per_model, out_path):
    n = len(per_model)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, per_model.items()):
        s, y = d["scores"], d["labels"]
        ax.hist(s[y == 0], bins=30, alpha=0.6, density=True, label="live")
        ax.hist(s[y == 1], bins=30, alpha=0.6, density=True, label="spoof")
        ax.axvline(0.5, color="k", ls="--", lw=1, label="thr=0.5")
        ax.axvline(d["m"]["eer_thr"], color="r", ls=":", lw=1.5,
                   label=f"EER thr={d['m']['eer_thr']:.3f}")
        ax.set_title(f"{name} — {domain}")
        ax.set_xlabel("P(spoof)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    models = {}
    for name, (ckpt_path, data_root) in CHECKPOINTS.items():
        if not os.path.isfile(ckpt_path):
            sys.exit(f"[error] checkpoint not found: {ckpt_path}")
        if not os.path.isdir(data_root):
            sys.exit(f"[error] data root not found: {data_root}")
        models[name] = load_checkpoint(build_model(), ckpt_path).to(DEVICE)
        print(f"Loaded {name:8s} <- {ckpt_path}")
        print(f"         eval on {data_root}")

    rows, results = [], {}
    for domain in TEST_DOMAINS:
        print(f"\n=== {domain} ===")
        results[domain] = {}
        for name, (_, data_root) in CHECKPOINTS.items():
            ddir = os.path.join(data_root, domain)
            if not os.path.isdir(ddir):
                print(f"  [skip] {ddir} not found")
                continue

            ds = DomainDataset(ddir, eval_tf)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
            s, y = get_scores(models[name], dl)
            mm = evaluate(s, y)
            results[domain][name] = dict(scores=s, labels=y, m=mm)

            print(f"  {name:8s}  AUC={mm['auc']:.4f}  EER={mm['eer']*100:5.2f}%  "
                  f"BPCER@APCER=1%: {mm['bpcer_sec']*100:5.2f}%")
            rows.append([
                domain, name, os.path.basename(data_root),
                f"{mm['auc']:.4f}", f"{mm['eer']*100:.2f}",
                f"{mm['acer_eer']*100:.2f}",
                f"{mm['bpcer_sec']*100:.2f}", f"{mm['apcer_sec']*100:.2f}",
                f"{mm['tau_sec']:.3f}",
                f"{mm['apcer50']*100:.2f}", f"{mm['bpcer50']*100:.2f}",
                f"{mm['acer50']*100:.2f}",
            ])

    for domain, per_model in results.items():
        if per_model:
            plot_roc(domain, per_model, os.path.join(OUT_DIR, f"roc_{domain}.png"))
            plot_scores(domain, per_model, os.path.join(OUT_DIR, f"scores_{domain}.png"))

    header = ["domain", "model", "eval_data", "AUC", "EER%", "ACER@EER%",
              "BPCER@APCER1%", "achieved_APCER%", "tau",
              "APCER@0.5%", "BPCER@0.5%", "ACER@0.5%"]

    with open(os.path.join(OUT_DIR, "comparison_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "comparison_results.md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")

    print(f"\nDone. Table + plots -> ./{OUT_DIR}/")
    print("Read AUC / EER / ACER@EER / BPCER@APCER=1%. The @0.5 columns are kept")
    print("for continuity but are measured at a threshold neither domain uses.")


if __name__ == "__main__":
    main()