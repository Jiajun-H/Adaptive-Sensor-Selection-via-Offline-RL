"""
Build offline RL transitions for IQL/CQL/BC training.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from artifact_utils import attach_artifact_signature
from config import Config
from dataset import generate_labels, load_and_clean, normalize
from model import MaskAwareRiskPredictor


class RandomPolicy:
    def __init__(self, n_sensors: int, m: int):
        self.n_sensors = n_sensors
        self.m = m
        self.name = "random"

    def select(self, t: int = 0) -> np.ndarray:
        mask = np.zeros(self.n_sensors, dtype=np.float32)
        chosen = np.random.choice(self.n_sensors, self.m, replace=False)
        mask[chosen] = 1.0
        return mask


class StaticTopMPolicy:
    def __init__(self, data_norm: np.ndarray, m: int):
        self.n_sensors = data_norm.shape[1]
        self.m = m
        self.name = "static_top_m"
        var = data_norm.var(axis=0)
        self.top_idx = np.argsort(var)[::-1][:m]
        self.fixed_mask = np.zeros(self.n_sensors, dtype=np.float32)
        self.fixed_mask[self.top_idx] = 1.0

    def select(self, t: int = 0) -> np.ndarray:
        return self.fixed_mask.copy()


class DynamicTopMPolicy:
    """Select sensors with the largest recent temporal changes."""

    def __init__(self, segments, m: int, window: int):
        first_seg = segments[0]["data_norm"]
        self.n_sensors = first_seg.shape[1]
        self.m = m
        self.window = max(2, int(window))
        self.name = "dynamic_top_m"
        self.current_data = first_seg

    def set_segment(self, segment) -> None:
        self.current_data = segment["data_norm"]

    def select(self, t: int = 0) -> np.ndarray:
        t = int(t)
        start = max(0, t - self.window + 1)
        window = self.current_data[start:t + 1]
        if len(window) >= 2:
            score = np.abs(window[-1] - window[0])
        else:
            score = np.abs(self.current_data[t])
        top_idx = np.argsort(score)[::-1][:self.m]
        mask = np.zeros(self.n_sensors, dtype=np.float32)
        mask[top_idx] = 1.0
        return mask


class ModalityCoveragePolicy:
    """Keep rare sensor modalities visible while filling the rest by variance."""

    def __init__(self, data_norm: np.ndarray, sensor_types, m: int):
        self.n_sensors = data_norm.shape[1]
        self.m = m
        self.name = "modality_coverage"
        self.sensor_types = list(sensor_types or ["unknown"] * self.n_sensors)
        if len(self.sensor_types) != self.n_sensors:
            self.sensor_types = ["unknown"] * self.n_sensors
        self.var_order = np.argsort(data_norm.var(axis=0))[::-1].tolist()
        self.groups = {}
        for idx, sensor_type in enumerate(self.sensor_types):
            self.groups.setdefault(sensor_type, []).append(idx)
        self.rare_types = [
            sensor_type
            for sensor_type, indices in sorted(self.groups.items(), key=lambda item: (len(item[1]), item[0]))
            if sensor_type not in {"temperature", "unknown"} and len(indices) > 0
        ]

    def select(self, t: int = 0) -> np.ndarray:
        chosen = []
        for sensor_type in self.rare_types:
            candidates = self.groups[sensor_type]
            chosen.append(candidates[int(np.random.randint(len(candidates)))])
            if len(chosen) >= self.m:
                break
        for idx in self.var_order:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= self.m:
                break
        mask = np.zeros(self.n_sensors, dtype=np.float32)
        mask[chosen[:self.m]] = 1.0
        return mask


class MixturePolicy:
    def __init__(self, random_policy: RandomPolicy, static_policy: StaticTopMPolicy):
        self.random_policy = random_policy
        self.static_policy = static_policy
        self.name = "mixture"

    def select(self, t: int = 0) -> np.ndarray:
        if np.random.random() < 0.5:
            return self.random_policy.select(t)
        return self.static_policy.select(t)


def build_window_inputs(obs_window: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim == 1:
        masks = masks[np.newaxis, :]
    obs_masked = obs_window[np.newaxis, :, :] * masks[:, np.newaxis, :]
    mask_expanded = np.repeat(masks[:, np.newaxis, :], obs_window.shape[0], axis=1)
    x = np.concatenate([obs_masked, mask_expanded], axis=-1)
    return torch.tensor(x, dtype=torch.float32)


def build_window_input(obs_window: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    return build_window_inputs(obs_window, mask)


def build_window_inputs_batch(obs_windows: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    """Build batched masked windows from aligned window/action rows."""
    obs_windows = np.asarray(obs_windows, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if obs_windows.ndim != 3:
        raise ValueError(f"obs_windows must be (B,W,N), got shape={obs_windows.shape}")
    if masks.ndim != 2:
        raise ValueError(f"masks must be (B,N), got shape={masks.shape}")
    if obs_windows.shape[0] != masks.shape[0] or obs_windows.shape[2] != masks.shape[1]:
        raise ValueError(f"window/mask shape mismatch: {obs_windows.shape} vs {masks.shape}")

    obs_masked = obs_windows * masks[:, np.newaxis, :]
    mask_expanded = np.broadcast_to(masks[:, np.newaxis, :], obs_windows.shape)
    x = np.concatenate([obs_masked, mask_expanded], axis=-1)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _encode_beliefs_in_chunks(
    model: MaskAwareRiskPredictor,
    x: torch.Tensor,
    batch_size: int,
    with_heads: bool = False,
):
    h_chunks = []
    logit_chunks = []
    tte_chunks = []
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size]
            h = model.encode_belief(xb)
            h_chunks.append(h.detach().cpu())
            if with_heads:
                logit_chunks.append(model.cls_head(h).squeeze(-1).detach().cpu())
                tte_chunks.append(model.tte_head(h).squeeze(-1).detach().cpu())

    h_all = torch.cat(h_chunks, dim=0) if h_chunks else torch.empty(0)
    if not with_heads:
        return h_all
    logits_all = torch.cat(logit_chunks, dim=0) if logit_chunks else torch.empty(0)
    tte_all = torch.cat(tte_chunks, dim=0) if tte_chunks else torch.empty(0)
    return h_all, logits_all, tte_all


def compute_reward(
    model: MaskAwareRiskPredictor,
    x: torch.Tensor,
    label: float,
    tte: float,
    mask: np.ndarray,
    prev_mask: np.ndarray,
    n_sensors: int,
    delta: int = 30,
    lambda_cost: float = 0.1,
    lambda_switch: float = 0.05,
    lambda_lead: float = 0.2,
) -> float:
    with torch.no_grad():
        logit, tte_hat = model(x)
        target_device = x.device
        target_label = torch.tensor([label], dtype=logit.dtype, device=target_device)
        target_tte = torch.tensor([tte], dtype=tte_hat.dtype, device=target_device)
        bce = F.binary_cross_entropy_with_logits(logit, target_label, reduction="mean")
        huber = F.huber_loss(tte_hat, target_tte, delta=1.0, reduction="mean")
        pred_loss = (bce + 0.5 * huber).item()
        prob = torch.sigmoid(logit).item()

    active_ratio = mask.sum() / n_sensors
    if prev_mask is not None:
        hamming = np.abs(mask - prev_mask).sum() / n_sensors
    else:
        hamming = 0.0

    lead_bonus = 0.0
    if label > 0.5 and tte > 0 and prob >= 0.5:
        ratio = min(tte / delta, 1.0)
        lead_bonus = (ratio ** 0.5) * 10.0
    elif label <= 0.5 and prob >= 0.5:
        lead_bonus = -1.0
    if label > 0.5 and prob < 0.5:
        lead_bonus = -2.0

    lambda_switch_adj = lambda_switch * 0.1
    reward = (
        -pred_loss
        - lambda_cost * active_ratio
        - lambda_switch_adj * hamming
        + lambda_lead * lead_bonus
    )
    return float(reward)


def compute_rewards_batch(
    logits: torch.Tensor,
    tte_hat: torch.Tensor,
    label: float,
    tte: float,
    masks: np.ndarray,
    prev_mask: np.ndarray,
    n_sensors: int,
    delta: int = 30,
    lambda_cost: float = 0.1,
    lambda_switch: float = 0.05,
    lambda_lead: float = 0.2,
    lambda_modality: float = 0.0,
    lambda_delta_risk: float = 0.0,
    sensor_type_weights: np.ndarray = None,
) -> np.ndarray:
    target_label = torch.full_like(logits, float(label))
    target_tte = torch.full_like(tte_hat, float(tte))
    bce = F.binary_cross_entropy_with_logits(logits, target_label, reduction="none")
    huber = F.huber_loss(tte_hat, target_tte, delta=1.0, reduction="none")
    pred_loss = (bce + 0.5 * huber).detach().cpu().numpy().astype(np.float32)
    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim == 1:
        masks = masks[np.newaxis, :]

    active_ratio = masks.mean(axis=1)
    hamming = np.zeros(len(masks), dtype=np.float32)
    if prev_mask is not None and len(masks) > 0:
        # Each sampled action at time t is an alternative decision. Penalize
        # switching from the previous real decision, not between alternatives.
        hamming = np.abs(masks - prev_mask[np.newaxis, :]).sum(axis=1) / n_sensors

    lead_bonus = np.zeros(len(masks), dtype=np.float32)
    if label > 0.5 and tte > 0:
        ratio = min(float(tte) / max(float(delta), 1.0), 1.0)
        lead_bonus[probs >= 0.5] = (ratio ** 0.5) * 10.0
        lead_bonus[probs < 0.5] = -2.0
    elif label <= 0.5:
        lead_bonus[probs >= 0.5] = -1.0

    lambda_switch_adj = lambda_switch * 0.1
    modality_bonus = np.zeros(len(masks), dtype=np.float32)
    if sensor_type_weights is not None and lambda_modality > 0:
        weights = np.asarray(sensor_type_weights, dtype=np.float32)
        if weights.shape[0] == masks.shape[1]:
            modality_bonus = (masks * weights[np.newaxis, :]).sum(axis=1) / np.maximum(masks.sum(axis=1), 1.0)
    delta_bonus = np.abs(probs - 0.5).astype(np.float32)
    rewards = (
        -pred_loss
        - lambda_cost * active_ratio
        - lambda_switch_adj * hamming
        + lambda_lead * lead_bonus
        + lambda_modality * modality_bonus
        + lambda_delta_risk * delta_bonus
    )
    return rewards.astype(np.float32)


def _prepare_segments(
    cfg: Config,
    data_norm,
    ttes,
    cls_labels,
    train_range,
    train_experiments,
    sensor_types=None,
):
    if train_experiments is not None and len(train_experiments) > 0:
        segments = []
        for exp in train_experiments:
            seg_data = exp["data_norm"]
            seg_cls = exp["cls_labels"]
            seg_tte = exp["ttes"]
            if len(seg_data) != len(seg_cls) or len(seg_data) != len(seg_tte):
                raise ValueError(f"Inconsistent experiment arrays: {exp.get('exp_id', 'unknown')}")
            segments.append(
                {
                    "exp_id": exp.get("exp_id", ""),
                    "data_norm": seg_data,
                    "cls_labels": seg_cls,
                    "ttes": seg_tte,
                    "sensor_types": exp.get("sensor_types", sensor_types or []),
                }
            )
        n_sensors = int(segments[0]["data_norm"].shape[1])
        for seg in segments:
            if seg["data_norm"].shape[1] != n_sensors:
                raise ValueError("All train experiments must have same sensor dimension.")
        print(f"[Build] multi-experiment mode, train experiments: {len(segments)}, sensors: {n_sensors}")
        return segments, n_sensors

    if train_range is None:
        train_range = (0, len(data_norm))
    t_start, t_end = int(train_range[0]), int(train_range[1])
    t_start = max(0, t_start)
    t_end = min(len(data_norm), t_end)
    if t_end <= t_start:
        raise ValueError(f"Invalid train_range: {train_range}")

    seg = {
        "exp_id": "legacy",
        "data_norm": data_norm[t_start:t_end],
        "cls_labels": cls_labels[t_start:t_end],
        "ttes": ttes[t_start:t_end],
        "sensor_types": sensor_types or getattr(cfg, "sensor_types", []),
    }
    n_sensors = int(seg["data_norm"].shape[1])
    print(f"[Build] single-segment mode, sensors: {n_sensors}, range: [{t_start}, {t_end})")
    return [seg], n_sensors


def build_offline_dataset(
    cfg: Config,
    data_norm=None,
    labels=None,
    ttes=None,
    cls_labels=None,
    n_sensors=None,
    train_range=None,
    train_experiments=None,
    predictor=None,
    sensor_types=None,
    save_name="offline_dataset.pt",
    artifact_signature=None,
):
    requested_device = str(getattr(cfg, "offline_device", getattr(cfg, "device", "cpu")))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[Build] device: {device}")

    if data_norm is None and (train_experiments is None or len(train_experiments) == 0):
        data_raw, cats, names = load_and_clean(
            cfg.data_path,
            auto_reverse=getattr(cfg, "auto_reverse_experiments", True),
        )
        t_steps = data_raw.shape[0]
        _labels, _ttes = generate_labels(
            data_raw,
            cats,
            names,
            temp_thresh=cfg.temp_threshold,
            trans_thresh=cfg.trans_threshold,
            heat_thresh=cfg.heat_threshold,
            co_thresh=cfg.co_threshold,
            min_modalities=cfg.hazard_min_modalities,
            min_consecutive=cfg.hazard_min_consecutive,
        )
        data_norm, _, _ = normalize(data_raw)
        labels = _labels
        ttes = _ttes
        cls_labels = np.zeros(t_steps, dtype=np.float32)
        for t in range(t_steps):
            if ttes[t] <= cfg.delta:
                cls_labels[t] = 1.0
        train_range = (0, t_steps)

    segments, n_sensors = _prepare_segments(
        cfg,
        data_norm=data_norm,
        ttes=ttes,
        cls_labels=cls_labels,
        train_range=train_range,
        train_experiments=train_experiments,
        sensor_types=sensor_types,
    )

    if predictor is not None:
        model = predictor.to(device)
        model.eval()
        print(f"[Build] using external predictor: {type(model).__name__}")
    else:
        ckpt_path = os.path.join(cfg.save_dir, "best_model.pt")
        assert os.path.exists(ckpt_path), f"Missing predictor checkpoint: {ckpt_path}"
        model = MaskAwareRiskPredictor(
            n_sensors=n_sensors,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            lstm_dropout=cfg.lstm_dropout,
            head_dropout=cfg.head_dropout,
        ).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        print("[Build] loaded predictor checkpoint")

    m = cfg.n_active if cfg.n_active > 0 else 20
    train_data_for_static = np.concatenate([seg["data_norm"] for seg in segments], axis=0)
    sensor_types = segments[0].get("sensor_types", []) or getattr(cfg, "sensor_types", [])
    if len(sensor_types) != n_sensors:
        sensor_types = ["unknown"] * n_sensors
    sensor_type_weights = np.zeros(n_sensors, dtype=np.float32)
    for idx, sensor_type in enumerate(sensor_types):
        if sensor_type not in {"temperature", "unknown"}:
            sensor_type_weights[idx] = 1.0
    random_pol = RandomPolicy(n_sensors, m)
    static_pol = StaticTopMPolicy(train_data_for_static, m)
    dynamic_pol = DynamicTopMPolicy(segments, m, cfg.window_size)
    modality_pol = ModalityCoveragePolicy(train_data_for_static, sensor_types, m)
    mixture_pol = MixturePolicy(random_pol, static_pol)
    policies = [random_pol, static_pol, dynamic_pol, modality_pol, mixture_pol]
    print(f"[Build] behavior policies: {[p.name for p in policies]}")
    if sensor_type_weights.sum() > 0:
        print(f"[Build] rare-modality reward sensors: {int(sensor_type_weights.sum())}/{n_sensors}")

    w = cfg.window_size
    k = cfg.iql_step_interval
    lambda_cost = cfg.iql_reward_lambda_cost
    lambda_switch = cfg.reward_lambda_switch
    lambda_lead = cfg.reward_lambda_lead
    lambda_modality = float(getattr(cfg, "reward_lambda_modality", 0.0))
    lambda_delta_risk = float(getattr(cfg, "reward_lambda_delta_risk", 0.0))
    n_mask = cfg.iql_n_mask_samples
    build_batch_steps = max(1, int(getattr(cfg, "offline_build_batch_steps", 64)))
    encode_batch_size = max(1, int(getattr(cfg, "offline_encode_batch_size", 4096)))

    decision_steps_total = 0
    for seg in segments:
        seg_len = len(seg["data_norm"])
        t_min = w - 1
        t_max = seg_len - k
        if t_max > t_min:
            decision_steps_total += len(list(range(t_min, t_max, k)))
    total = decision_steps_total * len(policies) * n_mask
    print(
        f"[Build] decision steps: {decision_steps_total}, "
        f"per-step samples: {n_mask}x{len(policies)}, total={total}"
    )
    print(
        f"[Build] batched encoding: steps_per_batch={build_batch_steps}, "
        f"encode_batch_size={encode_batch_size}"
    )

    transitions = []
    count = 0
    policy_stats = {}

    for pol in policies:
        pol_count = 0
        pol_rewards = []

        def flush_records(records):
            nonlocal count, pol_count
            if not records:
                return

            obs_t_rows = []
            obs_next_rows = []
            mask_rows = []
            for record in records:
                masks_local = record["masks"]
                obs_t_rows.extend([record["obs_t"]] * len(masks_local))
                obs_next_rows.extend([record["obs_next"]] * len(masks_local))
                mask_rows.append(masks_local)

            masks_flat = np.concatenate(mask_rows, axis=0).astype(np.float32, copy=False)
            obs_t_arr = np.stack(obs_t_rows, axis=0).astype(np.float32, copy=False)
            obs_next_arr = np.stack(obs_next_rows, axis=0).astype(np.float32, copy=False)

            x_t = build_window_inputs_batch(obs_t_arr, masks_flat).to(device, non_blocking=True)
            x_next = build_window_inputs_batch(obs_next_arr, masks_flat).to(device, non_blocking=True)

            h_t, logits_t, tte_hat_t = _encode_beliefs_in_chunks(
                model, x_t, encode_batch_size, with_heads=True
            )
            h_next = _encode_beliefs_in_chunks(
                model, x_next, encode_batch_size, with_heads=False
            )

            offset = 0
            for record in records:
                n_local = len(record["masks"])
                sl = slice(offset, offset + n_local)
                rewards = compute_rewards_batch(
                    logits_t[sl],
                    tte_hat_t[sl],
                    record["label"],
                    record["tte"],
                    record["masks"],
                    record["prev_mask"],
                    n_sensors,
                    cfg.delta,
                    lambda_cost,
                    lambda_switch,
                    lambda_lead,
                    lambda_modality,
                    lambda_delta_risk,
                    sensor_type_weights,
                )

                action_batch = torch.from_numpy(record["masks"].copy())
                reward_batch = torch.from_numpy(rewards.copy())
                done_batch = torch.full((n_local,), record["is_done"], dtype=torch.float32)
                for idx in range(n_local):
                    transitions.append(
                        {
                            "h": h_t[sl][idx],
                            "a": action_batch[idx],
                            "r": reward_batch[idx],
                            "h_next": h_next[sl][idx],
                            "done": done_batch[idx],
                        }
                    )
                    pol_rewards.append(float(rewards[idx]))
                    count += 1
                    pol_count += 1
                    if total > 0 and count % 2000 == 0:
                        print(f"  [{pol.name}] {count}/{total} ({100 * count / total:.1f}%)")
                offset += n_local

        for seg in segments:
            if hasattr(pol, "set_segment"):
                pol.set_segment(seg)
            seg_data = seg["data_norm"]
            seg_cls = seg["cls_labels"]
            seg_ttes = seg["ttes"]
            seg_len = len(seg_data)
            prev_mask = None
            records = []

            t_min = w - 1
            t_max = seg_len - k
            if t_max <= t_min:
                continue

            for t in range(t_min, t_max, k):
                t_next = min(t + k, seg_len - 1)
                is_done = 1.0 if t_next >= seg_len - 1 else 0.0

                obs_t = seg_data[t - w + 1:t + 1]
                obs_next = seg_data[t_next - w + 1:t_next + 1]
                label_t = seg_cls[t]
                tte_t = seg_ttes[t]

                masks = np.stack([pol.select(t) for _ in range(n_mask)], axis=0).astype(np.float32)
                records.append(
                    {
                        "obs_t": obs_t,
                        "obs_next": obs_next,
                        "label": float(label_t),
                        "tte": float(tte_t),
                        "masks": masks,
                        "prev_mask": None if prev_mask is None else prev_mask.copy(),
                        "is_done": float(is_done),
                    }
                )

                prev_mask = masks[-1]
                if len(records) >= build_batch_steps:
                    flush_records(records)
                    records = []

            flush_records(records)

        avg_r = float(np.mean(pol_rewards)) if pol_rewards else 0.0
        std_r = float(np.std(pol_rewards)) if pol_rewards else 0.0
        policy_stats[pol.name] = {
            "n_transitions": pol_count,
            "avg_reward": avg_r,
            "std_reward": std_r,
        }
        print(f"  [{pol.name}] done: {pol_count} transitions, avg_reward={avg_r:.4f}±{std_r:.4f}")

    if len(transitions) == 0:
        raise ValueError("No transitions generated. Check experiment length and window/step settings.")

    print(f"[Build] total transitions: {len(transitions)}")

    h_all = torch.stack([t["h"] for t in transitions])
    a_all = torch.stack([t["a"] for t in transitions])
    r_all = torch.stack([t["r"] for t in transitions])
    h_next_all = torch.stack([t["h_next"] for t in transitions])
    done_all = torch.stack([t["done"] for t in transitions])

    dataset_dict = attach_artifact_signature({
        "h": h_all,
        "a": a_all,
        "r": r_all,
        "h_next": h_next_all,
        "done": done_all,
        "n_sensors": n_sensors,
        "hidden_dim": cfg.hidden_dim,
        "policy_stats": policy_stats,
        "n_experiments": len(segments),
    }, artifact_signature)

    save_path = os.path.join(cfg.save_dir, save_name)
    torch.save(dataset_dict, save_path)
    print(f"[Build] saved: {save_path}")
    print(f"  h: {h_all.shape}, r: mean={r_all.mean():.4f} std={r_all.std():.4f}")
    return dataset_dict


class OfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.h = data["h"]
        self.a = data["a"]
        self.r = data["r"]
        self.h_next = data["h_next"]
        self.done = data["done"]
        self.n_sensors = data["n_sensors"]
        self.hidden_dim = data["hidden_dim"]

    def __len__(self):
        return len(self.h)

    def __getitem__(self, idx):
        return (self.h[idx], self.a[idx], self.r[idx], self.h_next[idx], self.done[idx])


if __name__ == "__main__":
    cfg = Config()
    if cfg.n_active == 0 or cfg.n_sensors == 0:
        data_raw, _, _ = load_and_clean(
            cfg.data_path,
            auto_reverse=getattr(cfg, "auto_reverse_experiments", True),
        )
        cfg.n_sensors = data_raw.shape[1]
        cfg.n_active = min(20, cfg.n_sensors)
        print(f"[Config] n_sensors={cfg.n_sensors}, n_active={cfg.n_active}")

    build_offline_dataset(cfg)
    print("offline dataset build completed.")
