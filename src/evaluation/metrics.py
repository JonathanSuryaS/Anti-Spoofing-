"""
src/evaluation/metrics.py
Standard face anti-spoofing benchmark metrics:
    - ACER  (Average Classification Error Rate) — primary metric
    - APCER (Attack Presentation Classification Error Rate)
    - BPCER (Bonafide Presentation Classification Error Rate)
    - EER   (Equal Error Rate)
    - HTER  (Half Total Error Rate) — used in cross-dataset eval
    - AUC   (Area Under ROC Curve)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_apcer_bpcer(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """
    APCER: fraction of spoof samples classified as real (attack errors)
    BPCER: fraction of real samples classified as spoof (bonafide errors)
    y_true: 1 = real, 0 = spoof
    y_pred: predicted probability of being real
    """
    spoof_mask = y_true == 0
    real_mask  = y_true == 1

    pred_labels = (y_pred >= threshold).astype(int)

    apcer = (pred_labels[spoof_mask] == 1).mean() if spoof_mask.any() else 0.0
    bpcer = (pred_labels[real_mask]  == 0).mean() if real_mask.any() else 0.0

    return float(apcer), float(bpcer)


def compute_acer(apcer: float, bpcer: float) -> float:
    """ACER = (APCER + BPCER) / 2 — the primary FAS benchmark metric."""
    return (apcer + bpcer) / 2.0


def compute_eer(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Equal Error Rate: threshold where FPR == FNR.
    Returns (eer, eer_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return eer, float(thresholds[eer_idx])


def compute_hter(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Half Total Error Rate — used in cross-dataset evaluation.
    HTER = (FAR + FRR) / 2
    """
    apcer, bpcer = compute_apcer_bpcer(y_true, y_pred, threshold)
    return (apcer + bpcer) / 2.0


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC score."""
    return float(roc_auc_score(y_true, y_pred))


def compute_all_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """
    Compute all FAS metrics at once.
    Args:
        y_true:      ground truth labels (1=real, 0=spoof)
        y_pred_prob: predicted probability of being real (softmax output[:,1])
    Returns:
        dict with all metrics
    """
    eer, eer_thresh = compute_eer(y_true, y_pred_prob)
    apcer, bpcer    = compute_apcer_bpcer(y_true, y_pred_prob, threshold=0.5)
    acer            = compute_acer(apcer, bpcer)
    hter            = compute_hter(y_true, y_pred_prob, threshold=0.5)
    auc             = compute_auc(y_true, y_pred_prob)

    metrics = {
        "acer":       round(acer  * 100, 2),   # in %
        "apcer":      round(apcer * 100, 2),
        "bpcer":      round(bpcer * 100, 2),
        "eer":        round(eer   * 100, 2),
        "hter":       round(hter  * 100, 2),
        "auc":        round(auc   * 100, 2),
        "eer_thresh": round(eer_thresh, 4),
    }

    return metrics
