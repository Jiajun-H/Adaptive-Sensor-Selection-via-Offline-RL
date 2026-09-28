"""
Evaluation utilities for predictor and policy-based sensor selection.
"""

import os
from typing import Dict, List, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(labels) == 0 or len(probs) == 0:
        return 0.0
    if np.unique(labels).size < 2:
        return 0.0
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.0


@torch.no_grad()
def collect_predictor_outputs(model, dataloader, device: str = "cuda") -> Dict[str, np.ndarray]:
    model.eval()
    all_logits, all_ttes_pred = [], []
    all_labels, all_ttes_true = [], []

    for x, label, tte_true in dataloader:
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        tte_true = tte_true.to(device, non_blocking=True)

        logit, tte_hat = model(x)
        all_logits.append(logit.cpu())
        all_ttes_pred.append(tte_hat.cpu())
        all_labels.append(label.cpu())
        all_ttes_true.append(tte_true.cpu())

    if len(all_logits) == 0:
        empty = np.array([], dtype=np.float32)
        return {
            "logits": empty,
            "probs": empty,
            "labels": empty,
            "ttes_pred": empty,
            "ttes_true": empty,
        }

    logits = torch.cat(all_logits).numpy().astype(np.float32)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    ttes_pred = torch.cat(all_ttes_pred).numpy().astype(np.float32)
    labels = torch.cat(all_labels).numpy().astype(np.float32)
    ttes_true = torch.cat(all_ttes_true).numpy().astype(np.float32)

    return {
        "logits": logits,
        "probs": probs,
        "labels": labels,
        "ttes_pred": ttes_pred,
        "ttes_true": ttes_true,
    }


@torch.no_grad()
def evaluate(model, criterion, dataloader, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    all_logits, all_ttes_pred = [], []
    all_labels, all_ttes_true = [], []
    total_loss, total_cls, total_tte = 0.0, 0.0, 0.0
    n_batches = 0

    for x, label, tte_true in dataloader:
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        tte_true = tte_true.to(device, non_blocking=True)

        logit, tte_hat = model(x)
        loss, cls_l, tte_l = criterion(logit, tte_hat, label, tte_true)

        total_loss += float(loss.item())
        total_cls += float(cls_l.item())
        total_tte += float(tte_l.item())
        n_batches += 1

        all_logits.append(logit.cpu())
        all_ttes_pred.append(tte_hat.cpu())
        all_labels.append(label.cpu())
        all_ttes_true.append(tte_true.cpu())

    if n_batches == 0:
        return {"loss": float("inf")}

    logits = torch.cat(all_logits).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    ttes_pred = torch.cat(all_ttes_pred).numpy()
    labels = torch.cat(all_labels).numpy()
    ttes_true = torch.cat(all_ttes_true).numpy()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = _safe_roc_auc(labels, probs)

    tte_mae = float(np.mean(np.abs(ttes_pred - ttes_true)))
    tte_rmse = float(np.sqrt(np.mean((ttes_pred - ttes_true) ** 2)))

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "tte_loss": total_tte / n_batches,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "tte_mae": tte_mae,
        "tte_rmse": tte_rmse,
    }


def print_metrics(metrics: Dict[str, float], prefix: str = "Val"):
    print(
        f"  [{prefix}] loss={metrics['loss']:.4f}  "
        f"cls={metrics['cls_loss']:.4f}  tte={metrics['tte_loss']:.4f}"
    )
    print(
        f"  [{prefix}] Acc={metrics['accuracy']:.4f}  "
        f"Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"AUC={metrics['auc']:.4f}"
    )
    print(
        f"  [{prefix}] TTE_MAE={metrics['tte_mae']:.2f}s  "
        f"TTE_RMSE={metrics['tte_rmse']:.2f}s"
    )


def detailed_report(model, dataloader, device: str = "cuda", threshold: float = 0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, label, _ in dataloader:
            x = x.to(device, non_blocking=True)
            logit, _ = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            all_probs.append(prob)
            all_labels.append(label.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds)
    print("\n=== Confusion Matrix ===")
    print(f"  TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
    print(f"  FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")

    fp_rate = cm[0, 1] / max(cm[0].sum(), 1)
    fn_rate = cm[1, 0] / max(cm[1].sum(), 1)
    print(f"  FPR = {fp_rate:.4f}")
    print(f"  FNR = {fn_rate:.4f}")


def compute_lead_time(
    probs: np.ndarray,
    ttes: np.ndarray,
    fire_onset: int,
    threshold: float = 0.5,
) -> dict:
    alarm_steps = np.where(probs >= threshold)[0]
    if len(alarm_steps) == 0:
        return {
            "avg_lead": 0.0,
            "median_lead": 0.0,
            "lead_ge_30s_ratio": 0.0,
            "first_alarm_step": -1,
            "n_alarms": 0,
        }

    first_alarm = int(alarm_steps[0])
    lead_times = ttes[alarm_steps]
    lead_times = lead_times[lead_times > 0]

    if len(lead_times) == 0:
        avg_lead = 0.0
        median_lead = 0.0
        lead_ge_30 = 0.0
    else:
        avg_lead = float(np.mean(lead_times))
        median_lead = float(np.median(lead_times))
        lead_ge_30 = float(np.mean(lead_times >= 30.0))

    return {
        "avg_lead": avg_lead,
        "median_lead": median_lead,
        "lead_ge_30s_ratio": lead_ge_30,
        "first_alarm_step": first_alarm,
        "first_alarm_tte": float(ttes[first_alarm]) if first_alarm < len(ttes) else 0.0,
        "n_alarms": int(len(alarm_steps)),
    }


def compute_switching_frequency(masks_sequence: np.ndarray) -> float:
    if len(masks_sequence) < 2:
        return 0.0
    jaccard_dists: List[float] = []
    for i in range(len(masks_sequence) - 1):
        a = masks_sequence[i]
        b = masks_sequence[i + 1]
        intersection = float(np.sum(np.minimum(a, b)))
        union = float(np.sum(np.maximum(a, b)))
        if union == 0:
            jd = 0.0
        else:
            jd = 1.0 - intersection / union
        jaccard_dists.append(jd)
    return float(np.mean(jaccard_dists))


def compute_event_miss_rate(
    probs: np.ndarray,
    cls_labels: np.ndarray,
    threshold: float = 0.5,
    min_event_gap: int = 10,
) -> dict:
    events = []
    in_event = False
    start = 0
    for i in range(len(cls_labels)):
        if cls_labels[i] == 1 and not in_event:
            start = i
            in_event = True
        elif cls_labels[i] == 0 and in_event:
            events.append((start, i))
            in_event = False
    if in_event:
        events.append((start, len(cls_labels)))

    if len(events) == 0:
        return {"n_events": 0, "n_missed": 0, "miss_rate": 0.0}

    merged = [events[0]]
    for s, e in events[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < min_event_gap:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    events = merged

    n_missed = 0
    for s, e in events:
        event_probs = probs[s:e]
        if np.all(event_probs < threshold):
            n_missed += 1

    n_events = len(events)
    miss_rate = n_missed / n_events if n_events > 0 else 0.0
    return {"n_events": n_events, "n_missed": n_missed, "miss_rate": miss_rate}


def _classification_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    if len(probs) == 0 or len(labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
        }
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = _safe_roc_auc(labels, probs)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }


def select_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
    default_threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Select the best decision threshold on validation predictions.

    The search is intentionally simple and stable: evaluate a fixed threshold
    grid, maximize the requested metric, and break ties by preferring the
    threshold closest to 0.5.
    """
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    if len(probs) == 0 or len(labels) == 0:
        base_metrics = _classification_from_probs(probs, labels, default_threshold)
        return {
            "threshold": float(default_threshold),
            "score": float(base_metrics.get(metric, 0.0)),
            "metric": metric,
            "metrics": base_metrics,
        }

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    if thresholds.size == 0:
        thresholds = np.array([default_threshold], dtype=np.float32)

    best_threshold = float(default_threshold)
    best_metrics = _classification_from_probs(probs, labels, best_threshold)
    best_score = float(best_metrics.get(metric, best_metrics.get("f1", 0.0)))

    for thr in thresholds:
        thr_val = float(thr)
        metrics = _classification_from_probs(probs, labels, thr_val)
        score = float(metrics.get(metric, metrics.get("f1", 0.0)))
        if score > best_score + 1e-12:
            best_threshold = thr_val
            best_metrics = metrics
            best_score = score
            continue
        if abs(score - best_score) <= 1e-12:
            if abs(thr_val - default_threshold) < abs(best_threshold - default_threshold):
                best_threshold = thr_val
                best_metrics = metrics
                best_score = score

    return {
        "threshold": float(best_threshold),
        "score": float(best_score),
        "metric": metric,
        "metrics": best_metrics,
    }


def evaluate_with_policy(
    predictor,
    policy_fn,
    data_norm: np.ndarray,
    cls_labels: np.ndarray,
    ttes: np.ndarray,
    fire_onset: int,
    cfg,
    device: str = "cpu",
) -> dict:
    predictor.eval()
    reset_policy = getattr(policy_fn, "reset", None)
    if callable(reset_policy):
        reset_policy()
    w = cfg.window_size
    t_steps = len(data_norm)
    n_sensors = data_norm.shape[1]
    m = cfg.n_active if cfg.n_active > 0 else 20

    all_probs = []
    all_labels = []
    all_ttes = []
    all_masks = []
    batch_select = getattr(policy_fn, "batch_select", None)

    if callable(batch_select):
        eval_batch_size = max(1, int(getattr(cfg, "policy_eval_batch_size", 2048)))
        obs_windows = []
        labels_rows = []
        ttes_rows = []
        for t in range(w - 1, t_steps):
            obs_windows.append(data_norm[t - w + 1:t + 1])
            labels_rows.append(float(cls_labels[t]))
            ttes_rows.append(float(ttes[t]))

        for start in range(0, len(obs_windows), eval_batch_size):
            obs_batch = np.asarray(obs_windows[start:start + eval_batch_size], dtype=np.float32)
            if len(obs_batch) == 0:
                continue
            masks = np.asarray(batch_select(obs_batch, n_sensors, m, device), dtype=np.float32)
            if masks.ndim != 2 or masks.shape != (len(obs_batch), n_sensors):
                raise ValueError(
                    f"batch_select must return shape {(len(obs_batch), n_sensors)}, got {masks.shape}"
                )

            obs_t = torch.from_numpy(obs_batch).to(device, non_blocking=True)
            masks_t = torch.from_numpy(masks).to(device, non_blocking=True)
            obs_masked = obs_t * masks_t.unsqueeze(1)
            mask_exp = masks_t.unsqueeze(1).expand(-1, w, -1)
            x_t = torch.cat([obs_masked, mask_exp], dim=-1)

            with torch.no_grad():
                p_hat, _ = predictor.predict(x_t)

            all_probs.extend(p_hat.detach().cpu().numpy().astype(np.float32).tolist())
            all_labels.extend(labels_rows[start:start + len(obs_batch)])
            all_ttes.extend(ttes_rows[start:start + len(obs_batch)])
            all_masks.extend([mask.copy() for mask in masks])
    else:
        for t in range(w - 1, t_steps):
            obs_window = data_norm[t - w + 1:t + 1]
            mask = policy_fn(predictor, obs_window, n_sensors, m, device)
            obs_masked = obs_window * mask[np.newaxis, :]
            mask_exp = np.tile(mask, (w, 1))
            x = np.concatenate([obs_masked, mask_exp], axis=-1)
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                p_hat, _ = predictor.predict(x_t)

            all_probs.append(float(p_hat.item()))
            all_labels.append(float(cls_labels[t]))
            all_ttes.append(float(ttes[t]))
            all_masks.append(mask.copy())

    if len(all_probs) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "lead_time": {
                "avg_lead": 0.0,
                "median_lead": 0.0,
                "lead_ge_30s_ratio": 0.0,
                "first_alarm_step": -1,
                "n_alarms": 0,
            },
            "switching_freq": 0.0,
            "event_miss_rate": {"n_events": 0, "n_missed": 0, "miss_rate": 0.0},
            "probs": np.array([], dtype=np.float32),
            "labels": np.array([], dtype=np.float32),
            "ttes": np.array([], dtype=np.float32),
            "masks": np.zeros((0, n_sensors), dtype=np.float32),
        }

    probs = np.array(all_probs, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.float32)
    ttes_arr = np.array(all_ttes, dtype=np.float32)
    masks_arr = np.array(all_masks, dtype=np.float32)

    cls_metrics = _classification_from_probs(probs, labels_arr, cfg.alarm_threshold)
    lead_info = compute_lead_time(probs, ttes_arr, fire_onset, cfg.alarm_threshold)
    switch_freq = compute_switching_frequency(masks_arr)
    event_miss = compute_event_miss_rate(probs, labels_arr, cfg.alarm_threshold)

    return {
        **cls_metrics,
        "lead_time": lead_info,
        "switching_freq": float(switch_freq),
        "event_miss_rate": event_miss,
        "probs": probs,
        "labels": labels_arr,
        "ttes": ttes_arr,
        "masks": masks_arr,
    }


def evaluate_with_policy_multi_experiments(
    predictor,
    policy_fn,
    experiments: List[dict],
    cfg,
    device: str = "cpu",
) -> dict:
    """
    Evaluate policy on multiple experiments and aggregate by concatenation.
    """
    if len(experiments) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "lead_time": {
                "avg_lead": 0.0,
                "median_lead": 0.0,
                "lead_ge_30s_ratio": 0.0,
                "first_alarm_step": -1,
                "n_alarms": 0,
            },
            "switching_freq": 0.0,
            "event_miss_rate": {"n_events": 0, "n_missed": 0, "miss_rate": 0.0},
            "probs": np.array([], dtype=np.float32),
            "labels": np.array([], dtype=np.float32),
            "ttes": np.array([], dtype=np.float32),
            "masks": np.zeros((0, 0), dtype=np.float32),
            "per_experiment": [],
        }

    per_experiment = []
    probs_list = []
    labels_list = []
    ttes_list = []
    masks_list = []
    total_event_n = 0
    total_event_missed = 0
    switch_sum = 0.0
    switch_count = 0

    for exp in experiments:
        reset_policy = getattr(policy_fn, "reset", None)
        if callable(reset_policy):
            reset_policy()
        fire_onset = int(exp.get("fire_onset", 0))
        m = evaluate_with_policy(
            predictor,
            policy_fn,
            exp["data_norm"],
            exp["cls_labels"],
            exp["ttes"],
            fire_onset,
            cfg,
            device,
        )
        per_experiment.append({"exp_id": exp.get("exp_id", ""), "metrics": m})
        probs_list.append(m["probs"])
        labels_list.append(m["labels"])
        ttes_list.append(m["ttes"])
        masks_list.append(m["masks"])

        em = m.get("event_miss_rate", {})
        total_event_n += int(em.get("n_events", 0))
        total_event_missed += int(em.get("n_missed", 0))

        masks = m["masks"]
        if len(masks) >= 2:
            local_switch = compute_switching_frequency(masks)
            switch_sum += float(local_switch) * float(len(masks) - 1)
            switch_count += int(len(masks) - 1)

    probs = np.concatenate(probs_list) if len(probs_list) > 0 else np.array([], dtype=np.float32)
    labels = np.concatenate(labels_list) if len(labels_list) > 0 else np.array([], dtype=np.float32)
    ttes = np.concatenate(ttes_list) if len(ttes_list) > 0 else np.array([], dtype=np.float32)
    masks = np.concatenate(masks_list, axis=0) if len(masks_list) > 0 else np.zeros((0, 0), dtype=np.float32)

    cls_metrics = _classification_from_probs(probs, labels, cfg.alarm_threshold)
    fire_onset_concat = int(np.where(labels > 0.5)[0][0]) if np.any(labels > 0.5) else int(len(labels))
    lead_info = compute_lead_time(probs, ttes, fire_onset_concat, cfg.alarm_threshold)
    event_miss = {
        "n_events": int(total_event_n),
        "n_missed": int(total_event_missed),
        "miss_rate": float(total_event_missed / total_event_n) if total_event_n > 0 else 0.0,
    }
    switch_freq = float(switch_sum / switch_count) if switch_count > 0 else 0.0

    return {
        **cls_metrics,
        "lead_time": lead_info,
        "switching_freq": switch_freq,
        "event_miss_rate": event_miss,
        "probs": probs,
        "labels": labels,
        "ttes": ttes,
        "masks": masks,
        "per_experiment": per_experiment,
    }


def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history.get("train_loss", []), label="Train Loss")
        axes[0].plot(history.get("val_loss", []), label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Loss Curves")

        axes[1].plot(history.get("val_f1", []), label="Val F1")
        axes[1].plot(history.get("val_auc", []), label="Val AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].set_title("Classification Metrics")

        axes[2].plot(history.get("val_tte_mae", []), label="Val TTE MAE")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MAE (s)")
        axes[2].legend()
        axes[2].set_title("TTE Prediction Error")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Plot] saved training curves to {save_path}")
    except ImportError:
        print("[Plot] matplotlib not available, skip plotting.")
