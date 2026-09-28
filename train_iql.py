"""
Training loops for offline RL policies: IQL, CQL, and BC.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))
from artifact_utils import attach_artifact_signature
from build_offline_dataset import OfflineRLDataset
from config import Config
from iql_networks import QNetwork, VNetwork, PolicyNetwork, expectile_loss


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rl_loader_kwargs(cfg: Config) -> dict:
    num_workers = max(0, int(getattr(cfg, "rl_num_workers", 0) or 0))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(getattr(cfg, "rl_prefetch_factor", 2)))
    return kwargs


def _configure_cuda_runtime(cfg: Config, device: torch.device) -> bool:
    use_amp = bool(getattr(cfg, "use_amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if bool(getattr(cfg, "allow_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return use_amp


def _sample_random_topk_masks(
    batch_size: int,
    n_sensors: int,
    m: int,
    n_masks: int,
    device: torch.device,
) -> torch.Tensor:
    m = max(1, min(int(m), int(n_sensors)))
    scores = torch.rand(n_masks, batch_size, n_sensors, device=device)
    top_idx = scores.topk(k=m, dim=-1).indices
    masks = torch.zeros_like(scores)
    masks.scatter_(2, top_idx, 1.0)
    return masks


def _q_random_actions_chunked(
    q_net: QNetwork,
    h: torch.Tensor,
    random_actions: torch.Tensor,
    max_eval_batch: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    k_neg, batch_size, n_sensors = random_actions.shape
    h_flat = h.detach().unsqueeze(0).expand(k_neg, -1, -1).reshape(k_neg * batch_size, -1)
    a_flat = random_actions.reshape(k_neg * batch_size, n_sensors)
    max_eval_batch = max(1, int(max_eval_batch))
    q_chunks = []
    for start in range(0, h_flat.shape[0], max_eval_batch):
        with _autocast(device, use_amp):
            q_chunks.append(q_net(h_flat[start:start + max_eval_batch], a_flat[start:start + max_eval_batch]))
    return torch.cat(q_chunks, dim=0).view(k_neg, batch_size).transpose(0, 1)


def _new_scaler(use_amp: bool):
    return torch.amp.GradScaler("cuda", enabled=use_amp)


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def balanced_action_bce_with_logits(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Per-sample BCE with selected and unselected sensors contributing equally."""
    loss = F.binary_cross_entropy_with_logits(logits, actions, reduction="none")
    pos = actions
    neg = 1.0 - actions
    pos_loss = (loss * pos).sum(dim=-1) / pos.sum(dim=-1).clamp_min(1.0)
    neg_loss = (loss * neg).sum(dim=-1) / neg.sum(dim=-1).clamp_min(1.0)
    return pos_loss + neg_loss


class DeviceOfflineRLDataset:
    """Offline RL tensors kept on the target device for GPU-side minibatching."""

    def __init__(self, dataset: OfflineRLDataset, device: torch.device):
        self.h = dataset.h.to(device, non_blocking=True)
        self.a = dataset.a.to(device, non_blocking=True)
        self.r = dataset.r.to(device, non_blocking=True)
        self.h_next = dataset.h_next.to(device, non_blocking=True)
        self.done = dataset.done.to(device, non_blocking=True)
        self.n_sensors = dataset.n_sensors
        self.hidden_dim = dataset.hidden_dim
        self.device = device

    def __len__(self):
        return int(self.h.shape[0])

    def iter_batches(self, batch_size: int, drop_last: bool = True, indices=None):
        base_indices = (
            torch.arange(len(self), device=self.device)
            if indices is None
            else torch.as_tensor(indices, device=self.device, dtype=torch.long)
        )
        n = int(base_indices.numel())
        if n == 0:
            return
        if drop_last:
            n_use = (n // batch_size) * batch_size
        else:
            n_use = n
        if n_use <= 0:
            return
        perm = base_indices[torch.randperm(n, device=self.device)[:n_use]]
        for start in range(0, n_use, batch_size):
            idx = perm[start:start + batch_size]
            if drop_last and idx.numel() < batch_size:
                continue
            yield self.h[idx], self.a[idx], self.r[idx], self.h_next[idx], self.done[idx]


def _split_rl_indices(n_samples: int, cfg: Config, done: Optional[torch.Tensor] = None):
    """Create a deterministic group split, preserving complete trajectories."""
    ratio = float(getattr(cfg, "rl_val_ratio", 0.1))
    if n_samples < 2 or ratio <= 0.0:
        return torch.arange(n_samples), torch.empty(0, dtype=torch.long)
    ratio = min(ratio, 0.5)
    generator = torch.Generator().manual_seed(int(getattr(cfg, "rl_val_seed", 2026)))

    if done is not None and int(done.numel()) == n_samples:
        terminal = done.detach().cpu().reshape(-1) > 0.5
        # Multiple masks at the final decision step produce consecutive done=1
        # rows. Only the last row of each run is a trajectory boundary.
        next_terminal = torch.zeros_like(terminal)
        next_terminal[:-1] = terminal[1:]
        group_ends = torch.where(terminal & ~next_terminal)[0].tolist()
        if not group_ends or group_ends[-1] != n_samples - 1:
            group_ends.append(n_samples - 1)
        group_starts = [0] + [end + 1 for end in group_ends[:-1]]
        groups = [
            torch.arange(start, end + 1, dtype=torch.long)
            for start, end in zip(group_starts, group_ends)
            if end >= start
        ]
        if len(groups) >= 2:
            group_order = torch.randperm(len(groups), generator=generator).tolist()
            target = max(1, int(round(n_samples * ratio)))
            val_group_ids = []
            val_size = 0
            for group_id in group_order[:-1]:
                val_group_ids.append(group_id)
                val_size += len(groups[group_id])
                if val_size >= target:
                    break
            val_group_set = set(val_group_ids)
            val_indices = torch.cat([groups[index] for index in val_group_ids])
            train_indices = torch.cat([
                group for index, group in enumerate(groups) if index not in val_group_set
            ])
            return train_indices, val_indices

    n_val = min(n_samples - 1, max(1, int(round(n_samples * ratio))))
    order = torch.randperm(n_samples, generator=generator)
    return order[n_val:], order[:n_val]


def _prepare_rl_split(dataset, cfg: Config, device: torch.device, requested_batch_size: int):
    train_indices, val_indices = _split_rl_indices(len(dataset), cfg, done=dataset.done)
    batch_size = min(max(1, int(requested_batch_size)), max(1, len(train_indices)))
    device_dataset = _maybe_preload_rl_dataset(dataset, cfg, device)
    loader = None
    train_device_indices = None
    if device_dataset is None:
        loader = DataLoader(
            Subset(dataset, train_indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **_rl_loader_kwargs(cfg),
        )
    else:
        train_device_indices = train_indices.to(device)
    print(
        f"[RL] trajectory split: train={len(train_indices)}, validation={len(val_indices)} "
        f"(seed={int(getattr(cfg, 'rl_val_seed', 2026))})"
    )
    return batch_size, val_indices, train_device_indices, device_dataset, loader


@torch.no_grad()
def _policy_validation_metrics(
    pi_net: PolicyNetwork,
    dataset: OfflineRLDataset,
    val_indices: torch.Tensor,
    cfg: Config,
    device: torch.device,
    q_net: Optional[QNetwork] = None,
):
    """Evaluate Top-M action agreement on transitions excluded from optimization."""
    if val_indices.numel() == 0:
        metrics = {
            "val_action_precision": float("nan"),
            "val_action_recall": float("nan"),
            "val_action_f1": float("nan"),
            "val_action_exact_match": float("nan"),
        }
        if q_net is not None:
            metrics["val_estimated_q"] = float("nan")
        return metrics

    max_samples = max(1, int(getattr(cfg, "rl_val_max_samples", 8192)))
    indices = val_indices[:max_samples]
    batch_size = min(max_samples, 4096)
    m_active = min(
        dataset.n_sensors,
        max(1, int(cfg.n_active if cfg.n_active > 0 else 20)),
    )
    pi_net.eval()
    if q_net is not None:
        q_net.eval()

    tp = fp = fn = exact = total = 0.0
    q_sum = 0.0
    q_count = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        h = dataset.h[idx].to(device, non_blocking=True)
        action = dataset.a[idx].to(device, non_blocking=True)
        logits = pi_net(h)
        top_idx = logits.topk(k=m_active, dim=-1).indices
        predicted = torch.zeros_like(logits)
        predicted.scatter_(1, top_idx, 1.0)

        tp += float((predicted * action).sum().item())
        fp += float((predicted * (1.0 - action)).sum().item())
        fn += float(((1.0 - predicted) * action).sum().item())
        exact += float((predicted == action).all(dim=1).sum().item())
        total += float(action.shape[0])
        if q_net is not None:
            q_values = q_net(h, predicted)
            q_sum += float(q_values.sum().item())
            q_count += int(q_values.numel())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    metrics = {
        "val_action_precision": precision,
        "val_action_recall": recall,
        "val_action_f1": f1,
        "val_action_exact_match": exact / max(total, 1.0),
    }
    if q_net is not None:
        metrics["val_estimated_q"] = q_sum / q_count if q_count else float("nan")
    return metrics


def _append_validation_history(history: dict, metrics: dict, epoch: int, updates: int, elapsed: float):
    history["epoch"].append(float(epoch))
    history["train_updates"].append(float(updates))
    history["epoch_time_s"].append(float(elapsed))
    for key, value in metrics.items():
        history[key].append(float(value))


def _maybe_preload_rl_dataset(dataset: OfflineRLDataset, cfg: Config, device: torch.device):
    if device.type != "cuda" or not bool(getattr(cfg, "rl_preload_to_device", True)):
        return None
    try:
        dev_ds = DeviceOfflineRLDataset(dataset, device)
        n_bytes = sum(t.numel() * t.element_size() for t in [dev_ds.h, dev_ds.a, dev_ds.r, dev_ds.h_next, dev_ds.done])
        print(f"[RL] preloaded offline dataset to {device} ({n_bytes / 1024 ** 2:.1f} MiB)")
        return dev_ds
    except RuntimeError as exc:
        print(f"[RL] preload to {device} failed, falling back to DataLoader: {exc}")
        return None


def train_iql(
    cfg: Config,
    epochs: Optional[int] = None,
    dataset_path: Optional[str] = None,
    save_path: Optional[str] = None,
    artifact_signature=None,
):
    epochs = epochs or cfg.iql_epochs
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = _configure_cuda_runtime(cfg, device)
    scaler = _new_scaler(use_amp)
    print(f"[IQL] device = {device}")
    print(f"[IQL] amp = {use_amp}")

    ds_path = dataset_path or os.path.join(cfg.save_dir, "offline_dataset.pt")
    assert os.path.exists(ds_path), f"Missing offline dataset: {ds_path}"

    dataset = OfflineRLDataset(ds_path)
    n_sensors = dataset.n_sensors
    hidden_dim = dataset.hidden_dim
    print(f"[IQL] dataset size: {len(dataset)}, hidden_dim={hidden_dim}, n_sensors={n_sensors}")
    print(f"[IQL] dataset path: {ds_path}")
    iql_batch_size, val_indices, train_device_indices, device_dataset, loader = _prepare_rl_split(
        dataset, cfg, device, cfg.iql_batch_size
    )

    q_net = QNetwork(hidden_dim, n_sensors, cfg.iql_hidden_dim).to(device)
    v_net = VNetwork(hidden_dim, cfg.iql_hidden_dim).to(device)
    pi_net = PolicyNetwork(hidden_dim, n_sensors, cfg.iql_hidden_dim).to(device)

    for name, net in [("Q", q_net), ("V", v_net), ("pi", pi_net)]:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  {name} params: {n_params:,}")

    opt_q = torch.optim.Adam(q_net.parameters(), lr=cfg.iql_lr)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=cfg.iql_lr)
    opt_pi = torch.optim.Adam(pi_net.parameters(), lr=cfg.iql_lr)

    tau = cfg.iql_tau
    gamma = cfg.iql_gamma
    alpha = cfg.iql_alpha
    weight_clip = cfg.iql_weight_clip
    grad_clip = cfg.iql_grad_clip

    best_val_f1 = float("-inf")
    patience_counter = 0
    history = {
        "epoch": [], "train_updates": [], "epoch_time_s": [],
        "v_loss": [], "q_loss": [], "pi_loss": [],
        "avg_reward": [], "train_logged_reward": [],
        "val_action_precision": [], "val_action_recall": [],
        "val_action_f1": [], "val_action_exact_match": [], "val_estimated_q": [],
    }
    cumulative_updates = 0

    print(f"[IQL] start training: epochs={epochs}, tau={tau}, gamma={gamma}, alpha={alpha}")
    print(f"      batch_size={iql_batch_size}, lr={cfg.iql_lr}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        q_net.train()
        v_net.train()
        pi_net.train()

        epoch_v_loss = 0.0
        epoch_q_loss = 0.0
        epoch_pi_loss = 0.0
        epoch_reward = 0.0
        n_batches = 0

        batch_iter = (
            device_dataset.iter_batches(
                iql_batch_size, drop_last=True, indices=train_device_indices
            )
            if device_dataset is not None
            else loader
        )

        for h, a, r, h_next, done in batch_iter:
            if device_dataset is None:
                h = h.to(device, non_blocking=True)
                a = a.to(device, non_blocking=True)
                r = r.to(device, non_blocking=True)
                h_next = h_next.to(device, non_blocking=True)
                done = done.to(device, non_blocking=True)

            with torch.no_grad():
                with _autocast(device, use_amp):
                    q_val = q_net(h, a)
            with _autocast(device, use_amp):
                v_val = v_net(h)
                v_loss = expectile_loss(q_val.float() - v_val.float(), tau)

            opt_v.zero_grad(set_to_none=True)
            scaler.scale(v_loss).backward()
            scaler.unscale_(opt_v)
            nn.utils.clip_grad_norm_(v_net.parameters(), grad_clip)
            scaler.step(opt_v)
            scaler.update()

            with torch.no_grad():
                with _autocast(device, use_amp):
                    v_next = v_net(h_next)
                    target_q = r + gamma * (1.0 - done) * v_next
            with _autocast(device, use_amp):
                q_pred = q_net(h, a)
                q_loss = F.mse_loss(q_pred, target_q)

            opt_q.zero_grad(set_to_none=True)
            scaler.scale(q_loss).backward()
            scaler.unscale_(opt_q)
            nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
            scaler.step(opt_q)
            scaler.update()

            with torch.no_grad():
                with _autocast(device, use_amp):
                    q_val_pi = q_net(h, a)
                    v_val_pi = v_net(h)
                advantage = q_val_pi.float() - v_val_pi.float()
                weight = torch.exp(advantage / alpha).clamp(max=weight_clip)

            with _autocast(device, use_amp):
                logits = pi_net(h)
                per_sample_bce = balanced_action_bce_with_logits(logits, a)
            pi_loss = (weight * per_sample_bce.float()).mean()

            opt_pi.zero_grad(set_to_none=True)
            scaler.scale(pi_loss).backward()
            scaler.unscale_(opt_pi)
            nn.utils.clip_grad_norm_(pi_net.parameters(), grad_clip)
            scaler.step(opt_pi)
            scaler.update()

            epoch_v_loss += float(v_loss.item())
            epoch_q_loss += float(q_loss.item())
            epoch_pi_loss += float(pi_loss.item())
            epoch_reward += float(r.mean().item())
            n_batches += 1

        avg_v = epoch_v_loss / max(n_batches, 1)
        avg_q = epoch_q_loss / max(n_batches, 1)
        avg_pi = epoch_pi_loss / max(n_batches, 1)
        avg_r = epoch_reward / max(n_batches, 1)

        history["v_loss"].append(avg_v)
        history["q_loss"].append(avg_q)
        history["pi_loss"].append(avg_pi)
        history["avg_reward"].append(avg_r)
        history["train_logged_reward"].append(avg_r)
        cumulative_updates += n_batches
        val_metrics = _policy_validation_metrics(
            pi_net, dataset, val_indices, cfg, device, q_net=q_net
        )
        _append_validation_history(
            history, val_metrics, epoch, cumulative_updates, time.perf_counter() - epoch_start
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"V={avg_v:.4f} Q={avg_q:.4f} pi={avg_pi:.4f} "
                f"ValF1={val_metrics['val_action_f1']:.4f} "
                f"ValQ={val_metrics['val_estimated_q']:.4f}"
            )

        current_val_f1 = float(val_metrics["val_action_f1"])
        if current_val_f1 > best_val_f1 + 1e-8:
            best_val_f1 = current_val_f1
            patience_counter = 0
            save_dict = attach_artifact_signature(
                {
                    "q_net": q_net.state_dict(),
                    "v_net": v_net.state_dict(),
                    "pi_net": pi_net.state_dict(),
                    "epoch": epoch,
                    "pi_loss": avg_pi,
                    "val_metrics": val_metrics,
                    "n_sensors": n_sensors,
                    "hidden_dim": hidden_dim,
                },
                artifact_signature,
            )
            iql_save_path = save_path or os.path.join(cfg.save_dir, "policy_best.pt")
            torch.save(save_dict, iql_save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.iql_patience:
                print(f"  early stop @ epoch {epoch} (patience={cfg.iql_patience})")
                break

    iql_save_path = save_path or os.path.join(cfg.save_dir, "policy_best.pt")
    print(f"[IQL] training done. best validation action F1 = {best_val_f1:.4f}")
    print(f"      saved to: {iql_save_path}")

    hist_path = os.path.join(cfg.save_dir, "iql_history.json")
    history_ser = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(hist_path, "w") as f:
        json.dump(history_ser, f, indent=2)
    print(f"      history saved to: {hist_path}")

    analyze_sensor_frequency(pi_net, dataset, cfg, device)
    return q_net, v_net, pi_net


def analyze_sensor_frequency(
    pi_net: PolicyNetwork,
    dataset: OfflineRLDataset,
    cfg: Config,
    device: torch.device,
):
    pi_net.eval()
    m = cfg.n_active if cfg.n_active > 0 else 20
    n_sensors = dataset.n_sensors

    freq = np.zeros(n_sensors, dtype=np.float64)
    overlap_sum = 0.0
    total_samples = min(len(dataset), 2000)

    loader = DataLoader(dataset, batch_size=256, shuffle=False, **_rl_loader_kwargs(cfg))
    count = 0

    with torch.no_grad():
        for h, a, r, h_next, done in loader:
            h = h.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            mask = pi_net.select_top_m(h, m=m)
            freq += mask.cpu().numpy().sum(axis=0)
            overlap_sum += float((mask * a).sum(dim=1).div(max(m, 1)).sum().item())
            count += h.shape[0]
            if count >= total_samples:
                break

    freq /= max(count, 1)
    avg_overlap = overlap_sum / max(count, 1)
    top_k = 10
    top_idx = np.argsort(freq)[::-1][:top_k]

    print(f"\n[analysis] sensor selection frequency Top-{top_k} ({count} samples, action_overlap={avg_overlap:.3f}):")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  #{rank:2d}  sensor {idx:3d}  freq = {freq[idx]:.4f}")


def train_cql(cfg: Config, epochs: Optional[int] = None, artifact_signature=None):
    epochs = epochs or cfg.cql_epochs
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = _configure_cuda_runtime(cfg, device)
    scaler = _new_scaler(use_amp)
    print(f"[CQL] device = {device}")
    print(f"[CQL] amp = {use_amp}")

    ds_path = os.path.join(cfg.save_dir, "offline_dataset.pt")
    assert os.path.exists(ds_path), f"Missing offline dataset: {ds_path}"

    dataset = OfflineRLDataset(ds_path)
    n_sensors = dataset.n_sensors
    hidden_dim = dataset.hidden_dim
    print(f"[CQL] dataset size: {len(dataset)}, hidden_dim={hidden_dim}, n_sensors={n_sensors}")
    cql_batch_size, val_indices, train_device_indices, device_dataset, loader = _prepare_rl_split(
        dataset,
        cfg,
        device,
        getattr(cfg, "cql_batch_size", cfg.iql_batch_size),
    )

    q_net = QNetwork(hidden_dim, n_sensors, cfg.iql_hidden_dim).to(device)
    q_target = QNetwork(hidden_dim, n_sensors, cfg.iql_hidden_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    for p in q_target.parameters():
        p.requires_grad_(False)

    pi_net = PolicyNetwork(hidden_dim, n_sensors, cfg.iql_hidden_dim).to(device)

    for name, net in [("Q", q_net), ("Q_target", q_target), ("pi", pi_net)]:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  {name} params: {n_params:,}")

    opt_q = torch.optim.Adam(q_net.parameters(), lr=cfg.cql_lr)
    opt_pi = torch.optim.Adam(pi_net.parameters(), lr=cfg.cql_lr)

    gamma = cfg.iql_gamma
    alpha_policy = cfg.iql_alpha
    weight_clip = cfg.iql_weight_clip
    grad_clip = cfg.iql_grad_clip
    cql_alpha = cfg.cql_alpha
    k_neg = max(1, int(cfg.cql_n_negative))
    max_negative_eval_batch = max(1, int(getattr(cfg, "cql_max_negative_eval_batch", 65536)))
    ema_tau = cfg.cql_target_ema
    m_active = cfg.n_active if cfg.n_active > 0 else 20

    best_val_f1 = float("-inf")
    patience_counter = 0
    history = {
        "epoch": [], "train_updates": [], "epoch_time_s": [],
        "q_loss": [], "cql_penalty": [], "pi_loss": [],
        "avg_reward": [], "train_logged_reward": [],
        "val_action_precision": [], "val_action_recall": [],
        "val_action_f1": [], "val_action_exact_match": [], "val_estimated_q": [],
    }
    cumulative_updates = 0

    print(f"[CQL] start training: epochs={epochs}, alpha_cql={cql_alpha}, K={k_neg}")
    print(
        f"      batch_size={cql_batch_size}, gamma={gamma}, alpha_policy={alpha_policy}, "
        f"ema_tau={ema_tau}, max_negative_eval_batch={max_negative_eval_batch}"
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        q_net.train()
        pi_net.train()

        epoch_q_loss = 0.0
        epoch_cql_pen = 0.0
        epoch_pi_loss = 0.0
        epoch_reward = 0.0
        n_batches = 0

        batch_iter = (
            device_dataset.iter_batches(
                cql_batch_size, drop_last=True, indices=train_device_indices
            )
            if device_dataset is not None
            else loader
        )

        for h, a, r, h_next, done in batch_iter:
            if device_dataset is None:
                h = h.to(device, non_blocking=True)
                a = a.to(device, non_blocking=True)
                r = r.to(device, non_blocking=True)
                h_next = h_next.to(device, non_blocking=True)
                done = done.to(device, non_blocking=True)
            batch_size = h.shape[0]

            with torch.no_grad():
                with _autocast(device, use_amp):
                    next_logits = pi_net(h_next)
                    next_top_idx = next_logits.topk(k=m_active, dim=-1).indices
                    next_a = torch.zeros_like(next_logits)
                    next_a.scatter_(1, next_top_idx, 1.0)
                    q_next = q_target(h_next, next_a)
                    target_q = r + gamma * (1.0 - done) * q_next

            random_actions = _sample_random_topk_masks(
                batch_size, n_sensors, m_active, k_neg, device
            )

            with _autocast(device, use_amp):
                q_data = q_net(h, a)
                bellman_loss = F.mse_loss(q_data, target_q)
            q_random_stack = _q_random_actions_chunked(
                q_net,
                h,
                random_actions,
                max_negative_eval_batch,
                device,
                use_amp,
            )
            with _autocast(device, use_amp):
                cql_penalty = (torch.logsumexp(q_random_stack, dim=1) - q_data).mean()
                q_loss = bellman_loss + cql_alpha * cql_penalty

            opt_q.zero_grad(set_to_none=True)
            scaler.scale(q_loss).backward()
            scaler.unscale_(opt_q)
            nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
            scaler.step(opt_q)
            scaler.update()

            with torch.no_grad():
                with _autocast(device, use_amp):
                    q_val = q_net(h, a)
                q_baseline = q_random_stack.mean(dim=1).float()
                advantage = q_val.float() - q_baseline
                weight = torch.exp(advantage / alpha_policy).clamp(max=weight_clip)

            with _autocast(device, use_amp):
                logits = pi_net(h)
                per_sample_bce = balanced_action_bce_with_logits(logits, a)
            pi_loss = (weight * per_sample_bce.float()).mean()

            opt_pi.zero_grad(set_to_none=True)
            scaler.scale(pi_loss).backward()
            scaler.unscale_(opt_pi)
            nn.utils.clip_grad_norm_(pi_net.parameters(), grad_clip)
            scaler.step(opt_pi)
            scaler.update()

            with torch.no_grad():
                for p_target, p_online in zip(q_target.parameters(), q_net.parameters()):
                    p_target.data.mul_(1.0 - ema_tau).add_(p_online.data, alpha=ema_tau)

            epoch_q_loss += float(q_loss.item())
            epoch_cql_pen += float(cql_penalty.item())
            epoch_pi_loss += float(pi_loss.item())
            epoch_reward += float(r.mean().item())
            n_batches += 1

        avg_q = epoch_q_loss / max(n_batches, 1)
        avg_cql = epoch_cql_pen / max(n_batches, 1)
        avg_pi = epoch_pi_loss / max(n_batches, 1)
        avg_r = epoch_reward / max(n_batches, 1)

        history["q_loss"].append(avg_q)
        history["cql_penalty"].append(avg_cql)
        history["pi_loss"].append(avg_pi)
        history["avg_reward"].append(avg_r)
        history["train_logged_reward"].append(avg_r)
        cumulative_updates += n_batches
        val_metrics = _policy_validation_metrics(
            pi_net, dataset, val_indices, cfg, device, q_net=q_net
        )
        _append_validation_history(
            history, val_metrics, epoch, cumulative_updates, time.perf_counter() - epoch_start
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Q={avg_q:.4f} CQL_pen={avg_cql:.4f} pi={avg_pi:.4f} "
                f"ValF1={val_metrics['val_action_f1']:.4f} "
                f"ValQ={val_metrics['val_estimated_q']:.4f}"
            )

        current_val_f1 = float(val_metrics["val_action_f1"])
        if current_val_f1 > best_val_f1 + 1e-8:
            best_val_f1 = current_val_f1
            patience_counter = 0
            save_dict = attach_artifact_signature(
                {
                    "q_net": q_net.state_dict(),
                    "q_target": q_target.state_dict(),
                    "pi_net": pi_net.state_dict(),
                    "epoch": epoch,
                    "pi_loss": avg_pi,
                    "val_metrics": val_metrics,
                    "n_sensors": n_sensors,
                    "hidden_dim": hidden_dim,
                },
                artifact_signature,
            )
            torch.save(save_dict, os.path.join(cfg.save_dir, "policy_cql.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.iql_patience:
                print(f"  early stop @ epoch {epoch} (patience={cfg.iql_patience})")
                break

    print(f"[CQL] training done. best validation action F1 = {best_val_f1:.4f}")
    print(f"      saved to: {os.path.join(cfg.save_dir, 'policy_cql.pt')}")

    hist_path = os.path.join(cfg.save_dir, "cql_history.json")
    history_ser = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(hist_path, "w") as f:
        json.dump(history_ser, f, indent=2)

    return q_net, pi_net


def train_bc(
    cfg: Config,
    epochs: Optional[int] = None,
    dataset_path: Optional[str] = None,
    save_path: Optional[str] = None,
    artifact_signature=None,
):
    epochs = epochs or cfg.bc_epochs
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = _configure_cuda_runtime(cfg, device)
    scaler = _new_scaler(use_amp)
    print(f"[BC] device = {device}")
    print(f"[BC] amp = {use_amp}")

    ds_path = dataset_path or os.path.join(cfg.save_dir, "offline_dataset.pt")
    assert os.path.exists(ds_path), f"Missing offline dataset: {ds_path}"

    dataset = OfflineRLDataset(ds_path)
    n_sensors = dataset.n_sensors
    hidden_dim = dataset.hidden_dim
    print(f"[BC] dataset size: {len(dataset)}, hidden_dim={hidden_dim}, n_sensors={n_sensors}")
    print(f"[BC] dataset path: {ds_path}")
    bc_batch_size, val_indices, train_device_indices, device_dataset, loader = _prepare_rl_split(
        dataset, cfg, device, cfg.iql_batch_size
    )

    pi_net = PolicyNetwork(hidden_dim, n_sensors, cfg.iql_hidden_dim).to(device)
    n_params = sum(p.numel() for p in pi_net.parameters())
    print(f"  pi params: {n_params:,}")

    opt_pi = torch.optim.Adam(pi_net.parameters(), lr=cfg.bc_lr)

    best_loss = float("inf")
    patience_counter = 0
    history = {
        "epoch": [], "train_updates": [], "epoch_time_s": [], "bc_loss": [],
        "val_action_precision": [], "val_action_recall": [],
        "val_action_f1": [], "val_action_exact_match": [],
    }
    cumulative_updates = 0

    print(f"[BC] start training: epochs={epochs}, lr={cfg.bc_lr}, batch_size={bc_batch_size}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        pi_net.train()
        epoch_loss = 0.0
        n_batches = 0

        batch_iter = (
            device_dataset.iter_batches(
                bc_batch_size, drop_last=True, indices=train_device_indices
            )
            if device_dataset is not None
            else loader
        )

        for h, a, r, h_next, done in batch_iter:
            if device_dataset is None:
                h = h.to(device, non_blocking=True)
                a = a.to(device, non_blocking=True)

            with _autocast(device, use_amp):
                logits = pi_net(h)
                bc_loss = balanced_action_bce_with_logits(logits, a).mean()

            opt_pi.zero_grad(set_to_none=True)
            scaler.scale(bc_loss).backward()
            scaler.unscale_(opt_pi)
            nn.utils.clip_grad_norm_(pi_net.parameters(), cfg.iql_grad_clip)
            scaler.step(opt_pi)
            scaler.update()

            epoch_loss += float(bc_loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["bc_loss"].append(avg_loss)
        cumulative_updates += n_batches
        val_metrics = _policy_validation_metrics(
            pi_net, dataset, val_indices, cfg, device
        )
        _append_validation_history(
            history, val_metrics, epoch, cumulative_updates, time.perf_counter() - epoch_start
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | BC Loss={avg_loss:.4f} "
                f"ValF1={val_metrics['val_action_f1']:.4f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_dict = attach_artifact_signature(
                {
                    "pi_net": pi_net.state_dict(),
                    "epoch": epoch,
                    "bc_loss": avg_loss,
                    "val_metrics": val_metrics,
                    "n_sensors": n_sensors,
                    "hidden_dim": hidden_dim,
                },
                artifact_signature,
            )
            bc_save_path = save_path or os.path.join(cfg.save_dir, "policy_bc.pt")
            torch.save(save_dict, bc_save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.iql_patience:
                print(f"  early stop @ epoch {epoch} (patience={cfg.iql_patience})")
                break

    bc_save_path = save_path or os.path.join(cfg.save_dir, "policy_bc.pt")
    print(f"[BC] training done. best loss = {best_loss:.4f}")
    print(f"      saved to: {bc_save_path}")

    hist_path = os.path.join(cfg.save_dir, "bc_history.json")
    history_ser = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(hist_path, "w") as f:
        json.dump(history_ser, f, indent=2)

    return pi_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IQL / CQL / BC policy training")
    parser.add_argument("--mode", type=str, default="iql", choices=["iql", "bc", "cql"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cql_batch_size", type=int, default=None)
    parser.add_argument("--cql_max_negative_eval_batch", type=int, default=None)
    parser.add_argument("--rl_val_ratio", type=float, default=None)
    parser.add_argument("--rl_val_max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_preload", action="store_true", help="Disable preloading offline RL tensors to CUDA.")
    args = parser.parse_args()

    cfg = Config()
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    _set_training_seed(args.seed)
    if args.lr is not None:
        cfg.iql_lr = args.lr
        cfg.bc_lr = args.lr
        cfg.cql_lr = args.lr
    if args.batch_size is not None:
        cfg.iql_batch_size = args.batch_size
        cfg.cql_batch_size = args.batch_size
    if args.cql_batch_size is not None:
        cfg.cql_batch_size = args.cql_batch_size
    if args.cql_max_negative_eval_batch is not None:
        cfg.cql_max_negative_eval_batch = args.cql_max_negative_eval_batch
    if args.rl_val_ratio is not None:
        cfg.rl_val_ratio = args.rl_val_ratio
    if args.rl_val_max_samples is not None:
        cfg.rl_val_max_samples = args.rl_val_max_samples
    if args.no_preload:
        cfg.rl_preload_to_device = False

    if args.mode == "bc":
        ep = args.epochs if args.epochs is not None else cfg.bc_epochs
        train_bc(cfg, epochs=ep)
        print("BC training completed.")
    elif args.mode == "cql":
        ep = args.epochs if args.epochs is not None else cfg.cql_epochs
        train_cql(cfg, epochs=ep)
        print("CQL training completed.")
    else:
        ep = args.epochs if args.epochs is not None else cfg.iql_epochs
        train_iql(cfg, epochs=ep)
        print("IQL training completed.")
