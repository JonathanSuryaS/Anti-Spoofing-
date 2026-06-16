"""
FAS Metrics — APCER / BPCER / ACER  (+ HTER, AUC, EER)
=====================================================
The metrics that matter for face anti-spoofing. Accuracy is deliberately
NOT the headline because test sets are imbalanced and per-class error is
what actually matters.

Definitions (live=0 = bona fide, spoof=1 = attack):
  APCER = fraction of ATTACKS wrongly accepted as live   (security risk)
  BPCER = fraction of LIVE  wrongly rejected as spoof     (usability risk)
  ACER  = (APCER + BPCER) / 2                              (overall)
  HTER  = half total error rate (common cross-dataset)    (== ACER at t)
  AUC   = area under ROC (threshold-independent)
  EER   = error rate where APCER == BPCER (threshold-independent)

Spoof "score" = softmax(logits)[:, spoof] in [0,1]. A threshold t turns
scores into decisions. APCER/BPCER/ACER depend on t; AUC/EER do not.
"""
import numpy as np


def apcer_bpcer_acer(scores, labels, threshold=0.5):
    scores = np.asarray(scores); labels = np.asarray(labels)
    pred_spoof = scores >= threshold
    attacks = labels == 1
    bonafide = labels == 0
    apcer = np.mean(~pred_spoof[attacks]) if attacks.sum() else 0.0
    bpcer = np.mean(pred_spoof[bonafide]) if bonafide.sum() else 0.0
    return apcer, bpcer, (apcer + bpcer) / 2.0


def compute_eer(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    best_t, best_gap, best_eer = 0.5, 1e9, 1.0
    for t in np.unique(scores):
        apcer, bpcer, _ = apcer_bpcer_acer(scores, labels, t)
        gap = abs(apcer - bpcer)
        if gap < best_gap:
            best_gap, best_eer, best_t = gap, (apcer + bpcer) / 2, t
    return best_eer, best_t


def compute_auc(scores, labels):
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def hter(scores, labels, threshold=0.5):
    return apcer_bpcer_acer(scores, labels, threshold)[2]


def summary(scores, labels, threshold=0.5):
    apcer, bpcer, acer = apcer_bpcer_acer(scores, labels, threshold)
    eer, _ = compute_eer(scores, labels)
    auc = compute_auc(scores, labels)
    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer, "EER": eer,
            "AUC": auc, "threshold": threshold,
            "n_live": int(np.sum(np.asarray(labels) == 0)),
            "n_spoof": int(np.sum(np.asarray(labels) == 1))}


def format_summary(name, s):
    return (f"{name:14s} | APCER {100*s['APCER']:5.2f}%  "
            f"BPCER {100*s['BPCER']:5.2f}%  ACER {100*s['ACER']:5.2f}%  "
            f"EER {100*s['EER']:5.2f}%  AUC {s['AUC']:.4f}  "
            f"(live {s['n_live']}, spoof {s['n_spoof']})")