"""Evaluation metrics (pure NumPy, unit-tested).

Frame level
    :func:`classification_report` - per-class precision / recall / F1 and confusion matrix.

Online / temporal level (what matters on a running printer)
    :func:`alarm_onsets` - turn per-frame scores into alarm events with k-of-n persistence.
    :func:`detection_delay` - frames (and seconds) from true defect onset to the first alarm.
    :func:`false_alarms` - alarms raised on defect-free frames, per 100 frames and per print-hour.

Detection mAP is computed by Ultralytics (``model.val``) and is not re-implemented here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ClassMetrics:
    """Precision / recall / F1 / support of one class."""

    precision: float
    recall: float
    f1: float
    support: int


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of samples of true class ``i`` predicted as ``j``."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict:
    """Per-class P/R/F1, macro averages, accuracy and the confusion matrix (JSON-ready)."""
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    per: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[name] = asdict(ClassMetrics(float(p), float(r), float(f), int(cm[i, :].sum())))
    n = int(cm.sum())
    return {
        "accuracy": float(np.trace(cm) / n) if n else 0.0,
        "macro_precision": float(np.mean([v["precision"] for v in per.values()])),
        "macro_recall": float(np.mean([v["recall"] for v in per.values()])),
        "macro_f1": float(np.mean([v["f1"] for v in per.values()])),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "n": n,
    }


def persistent(scores: Sequence[float], threshold: float, k: int = 3, n: int = 5) -> np.ndarray:
    """Boolean alarm state per frame: at least ``k`` of the last ``n`` scores >= ``threshold``.

    Causal (uses only past frames) - the same rule the live inference worker applies.
    """
    above = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    if k <= 1:
        return above.astype(bool)
    csum = np.concatenate([[0], np.cumsum(above)])
    idx = np.arange(1, len(above) + 1)
    window = csum[idx] - csum[np.maximum(0, idx - n)]
    return window >= k


def alarm_onsets(state: Sequence[bool]) -> list[int]:
    """Frame indices where the alarm state switches from off to on."""
    s = np.asarray(state, dtype=bool)
    if s.size == 0:
        return []
    prev = np.concatenate([[False], s[:-1]])
    return [int(i) for i in np.where(s & ~prev)[0]]


@dataclass(frozen=True)
class DelayResult:
    """Detection delay for one sequence with a known onset."""

    onset: int
    first_alarm: int | None
    delay_frames: int | None
    early_alarm: bool  # an alarm fired before the true onset

    @property
    def detected(self) -> bool:
        """True when an alarm fired at or after the onset."""
        return self.delay_frames is not None


def detection_delay(state: Sequence[bool], onset: int) -> DelayResult:
    """Frames from the true defect ``onset`` to the first alarm at or after it."""
    s = np.asarray(state, dtype=bool)
    early = bool(s[:onset].any())
    after = np.where(s[onset:])[0]
    if after.size == 0:
        return DelayResult(onset, None, None, early)
    first = int(onset + after[0])
    return DelayResult(onset, first, first - onset, early)


def false_alarms(states: Sequence[Sequence[bool]], labels: Sequence[Sequence[int]], seconds_per_frame: float) -> dict:
    """Count alarm onsets that start on defect-free frames (label 0).

    Returns the count, per 100 frames and per print-hour. ``seconds_per_frame`` converts frame
    counts to time; YOLO-Strings captures one image per layer, so the per-hour figure depends on
    this assumed layer time and is reported with it.
    """
    n_alarms = 0
    n_clean = 0
    for s, y in zip(states, labels, strict=True):
        y_arr = np.asarray(y)
        n_clean += int((y_arr == 0).sum())
        n_alarms += sum(1 for i in alarm_onsets(s) if y_arr[i] == 0)
    hours = n_clean * seconds_per_frame / 3600.0
    return {
        "false_alarms": n_alarms,
        "clean_frames": n_clean,
        "per_100_frames": 100.0 * n_alarms / n_clean if n_clean else 0.0,
        "per_print_hour": n_alarms / hours if hours else 0.0,
        "assumed_seconds_per_frame": seconds_per_frame,
    }


def summarize_delays(results: Sequence[DelayResult], seconds_per_frame: float) -> dict:
    """Aggregate delays over sequences (median / mean in frames and seconds, detection rate)."""
    detected = [r.delay_frames for r in results if r.delay_frames is not None]
    out = {
        "sequences": len(results),
        "detected": len(detected),
        "detection_rate": len(detected) / len(results) if results else 0.0,
        "early_alarm_sequences": sum(r.early_alarm for r in results),
    }
    if detected:
        out |= {
            "median_delay_frames": float(np.median(detected)),
            "mean_delay_frames": float(np.mean(detected)),
            "median_delay_s": float(np.median(detected) * seconds_per_frame),
        }
    return out


def binary_pr(scores: Sequence[float], labels: Sequence[int], threshold: float) -> dict:
    """Frame-level precision / recall / F1 for one threshold, ignoring labels < 0 (unknown)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    m = y >= 0
    pred, y = s[m] >= threshold, y[m] == 1
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the precision-recall curve (step interpolation), ignoring labels < 0."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    m = y >= 0
    s, y = s[m], y[m] == 1
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())
