"""
Dataset utilities for fire-risk modeling.

This module supports two modes:
1) Legacy single-file mode (data.xlsx / npy fallback).
2) Multi-experiment mode from a directory of .xlsx files.
"""

import os
from pathlib import Path
import re
import fnmatch
import warnings
from typing import Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler


# ======================================================================
# .npy loading (legacy)
# ======================================================================


def load_from_npy(data_dir: str) -> dict:
    full_data_path = os.path.join(data_dir, "full_data.npy")
    hazard_path = os.path.join(data_dir, "hazard.npy")
    y_labels_path = os.path.join(data_dir, "y_labels.npy")
    tte_labels_path = os.path.join(data_dir, "tte_labels.npy")

    data_norm = np.load(full_data_path).astype(np.float32)
    hazard = np.load(hazard_path).astype(np.float32)
    cls_labels = np.load(y_labels_path).astype(np.float32)
    ttes = np.load(tte_labels_path).astype(np.float32)

    t_steps, n_sensors = data_norm.shape
    danger_idx = np.where(hazard > 0.5)[0]
    fire_onset = int(danger_idx[0]) if len(danger_idx) > 0 else t_steps

    return {
        "data_norm": data_norm,
        "labels": hazard,
        "cls_labels": cls_labels,
        "ttes": ttes,
        "n_sensors": n_sensors,
        "fire_onset": fire_onset,
    }


def has_npy_data(data_dir: str) -> bool:
    required = ["full_data.npy", "hazard.npy", "y_labels.npy", "tte_labels.npy"]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required)


# ======================================================================
# helpers
# ======================================================================


def _is_missing_name(v) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def _safe_name(v) -> str:
    if _is_missing_name(v):
        return ""
    return str(v).strip()


def _safe_cat(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _to_numeric_col(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_header_text(v) -> str:
    s = _safe_name(v)
    if not s:
        return ""
    s = s.lower()
    replacements = {
        "（": "(",
        "）": ")",
        "／": "/",
        "·": "",
        "²": "2",
        "°c": "℃",
        " ": "",
        "\t": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


_TEMP_PROBE_RE = re.compile(r"^\d+-\d+(测点温度)?$")
_HEAT_PROBE_RE = re.compile(r"^\d+(\.\d+)?d$")


def _infer_sensor_type(cat: str, name: str) -> str:
    cat_n = _normalize_header_text(cat)
    name_n = _normalize_header_text(name)
    text = f"{cat_n}|{name_n}"

    if (
        "temperature" in text
        or "temp" in text
        or "温度" in text
        or "℃" in str(cat)
        or "℃" in str(name)
        or _TEMP_PROBE_RE.fullmatch(name_n) is not None
    ):
        return "temperature"
    if "透光" in text or "trans" in text:
        return "transmittance"
    if (
        "热流" in text
        or "辐射" in text
        or "通量" in text
        or "flux" in text
        or "kw/m" in text
        or _HEAT_PROBE_RE.fullmatch(name_n) is not None
    ):
        return "heat_flux"
    if "烟气层" in text or "smokelayer" in text or "smoke" in text:
        return "smoke_layer"
    if "co浓度" in text or cat_n.startswith("co") or name_n == "co":
        return "co"
    return "unknown"


def infer_sensor_types(cats: List[str], names: List[str]) -> List[str]:
    n_cols = max(len(cats), len(names))
    sensor_types: List[str] = []
    for j in range(n_cols):
        cat = cats[j] if j < len(cats) else ""
        name = names[j] if j < len(names) else ""
        sensor_types.append(_infer_sensor_type(cat, name))
    return sensor_types


def _group_sensor_indices(sensor_types: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, sensor_type in enumerate(sensor_types):
        groups.setdefault(sensor_type, []).append(idx)
    return groups


def _family_series(data_raw: np.ndarray, indices: List[int], sensor_type: str) -> np.ndarray:
    if len(indices) == 0:
        return np.array([], dtype=np.float32)
    values = data_raw[:, indices]
    if sensor_type == "transmittance":
        return np.nanmin(values, axis=1).astype(np.float32)
    return np.nanmax(values, axis=1).astype(np.float32)


def _trend_vote(series: np.ndarray, danger_increases: bool, base_margin: float) -> int:
    if len(series) < 6:
        return 0
    valid = series[np.isfinite(series)]
    if len(valid) < 6:
        return 0

    window = min(max(10, len(series) // 10), 60, len(series) // 2)
    if window < 3:
        return 0

    head = float(np.median(series[:window]))
    tail = float(np.median(series[-window:]))
    span = float(np.quantile(valid, 0.9) - np.quantile(valid, 0.1))
    margin = max(base_margin, 0.1 * span)

    delta = (tail - head) if danger_increases else (head - tail)
    if delta > margin:
        return 1
    if delta < -margin:
        return -1
    return 0


def _should_reverse_experiment(data_raw: np.ndarray, cats: List[str], names: List[str]) -> bool:
    sensor_types = infer_sensor_types(cats, names)
    groups = _group_sensor_indices(sensor_types)
    votes: List[int] = []

    for sensor_type, danger_increases, base_margin in [
        ("temperature", True, 5.0),
        ("co", True, 20.0),
        ("heat_flux", True, 0.2),
        ("transmittance", False, 0.05),
    ]:
        indices = groups.get(sensor_type, [])
        if len(indices) == 0:
            continue
        series = _family_series(data_raw, indices, sensor_type)
        vote = _trend_vote(series, danger_increases=danger_increases, base_margin=base_margin)
        if vote != 0:
            votes.append(vote)

    if len(votes) == 0:
        return False

    forward_votes = sum(1 for v in votes if v > 0)
    reverse_votes = sum(1 for v in votes if v < 0)
    return reverse_votes >= 2 and reverse_votes > forward_votes


def _first_sustained_true(mask: np.ndarray, min_consecutive: int) -> Optional[int]:
    run_len = 0
    start = 0
    for idx, is_true in enumerate(mask):
        if is_true:
            if run_len == 0:
                start = idx
            run_len += 1
            if run_len >= min_consecutive:
                return start
        else:
            run_len = 0
    return None


def _channel_keys(names: List[str]) -> List[str]:
    # Keep duplicate channel names distinguishable and stable by occurrence order.
    seen: Dict[str, int] = {}
    keys: List[str] = []
    for name in names:
        base = name if name else "unnamed"
        count = seen.get(base, 0)
        keys.append(f"{base}__{count}")
        seen[base] = count + 1
    return keys


def _first_positive_idx(arr: np.ndarray) -> int:
    idx = np.where(arr > 0.5)[0]
    return int(idx[0]) if len(idx) > 0 else int(len(arr))


def _concat_arrays(arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
    if len(arrays) == 0:
        return np.array([], dtype=np.float32)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=axis)


def _window_labels_from_dataset(ds: Dataset) -> np.ndarray:
    if isinstance(ds, FireDataset):
        idx = np.asarray(ds.indices, dtype=np.int64)
        return ds.cls_labels[idx].astype(np.float32)
    if isinstance(ds, ConcatDataset):
        return _concat_arrays([_window_labels_from_dataset(sub_ds) for sub_ds in ds.datasets], axis=0)
    if isinstance(ds, Subset):
        base_labels = _window_labels_from_dataset(ds.dataset)
        subset_idx = np.asarray(ds.indices, dtype=np.int64)
        return base_labels[subset_idx].astype(np.float32)
    raise TypeError(f"Unsupported dataset type for balanced sampling: {type(ds).__name__}")


def _build_loader(ds: Dataset, cfg, training: bool) -> DataLoader:
    sampler = None
    shuffle = training

    if training and getattr(cfg, "balanced_train_sampling", False):
        labels = _window_labels_from_dataset(ds)
        pos_mask = labels > 0.5
        n_pos = int(pos_mask.sum())
        n_neg = int(len(labels) - n_pos)

        if n_pos > 0 and n_neg > 0:
            weights = np.where(
                pos_mask,
                0.5 / max(n_pos, 1),
                0.5 / max(n_neg, 1),
            ).astype(np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
            print(f"[Data] balanced sampler enabled: windows pos/neg={n_pos}/{n_neg}")

    num_workers = max(0, int(getattr(cfg, "data_num_workers", 0) or 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(getattr(cfg, "data_prefetch_factor", 2)))

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=training,
        **loader_kwargs,
    )


def _list_experiment_files(data_dir: str, pattern: str) -> List[Path]:
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        return []

    pattern_l = pattern.lower()

    def _matches(p: Path) -> bool:
        return p.is_file() and fnmatch.fnmatch(p.name.lower(), pattern_l)

    files = sorted([p for p in root.iterdir() if _matches(p)])
    if len(files) == 0:
        files = sorted([p for p in root.rglob("*") if _matches(p)])
        if len(files) > 0:
            print(f"[Data] no top-level matches for {pattern!r}; using recursive scan under {root}")

    # Ignore Office lock/temp files
    files = [p for p in files if not p.name.startswith("~$")]
    return files


def _resolve_experiment_dir(data_dir: str) -> Optional[Path]:
    if not data_dir:
        return None

    configured = Path(data_dir)
    if configured.exists() and configured.is_dir():
        return configured

    repo_root = Path(__file__).resolve().parent
    target_name = configured.name.lower()
    candidates = sorted(
        [
            p for p in repo_root.rglob("*")
            if p.is_dir() and p.name.lower() == target_name
        ]
    )
    if len(candidates) > 0:
        print(f"[Data] configured data_dir not found: {configured}; using detected directory: {candidates[0]}")
        return candidates[0]

    return configured


def _read_raw_header_rows(data_path: str) -> Tuple[List[str], List[str], int]:
    df_raw = pd.read_excel(data_path, header=None)
    row_cat = df_raw.iloc[0].tolist() if len(df_raw) > 0 else []
    row_name = df_raw.iloc[1].tolist() if len(df_raw) > 1 else []
    n_cols = df_raw.shape[1]
    cats = [_safe_cat(v) for v in row_cat] + [""] * max(0, n_cols - len(row_cat))
    names = [_safe_name(v) for v in row_name] + [""] * max(0, n_cols - len(row_name))
    return cats[:n_cols], names[:n_cols], n_cols


# ======================================================================
# loading and preprocessing
# ======================================================================


def load_and_clean(
    data_path: str,
    broken_val: float = -200.0,
    broken_ratio: float = 0.5,
    name_template_by_col: Optional[List[str]] = None,
    auto_reverse: bool = False,
    preserve_missing: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load one Excel experiment and keep usable numeric sensor columns.

    `name_template_by_col` is used as fallback when a column name is missing.
    When `preserve_missing` is true, NaN and the broken-value sentinel remain
    missing so a caller can fit imputation on its training partition only.
    """
    df_raw = pd.read_excel(data_path, header=None)
    row_cat = df_raw.iloc[0]
    row_name = df_raw.iloc[1]
    df_data = df_raw.iloc[2:].reset_index(drop=True)

    valid_idx: List[int] = []
    cats: List[str] = []
    names: List[str] = []

    skip_exact = {"时间", "TTime", "设备地址", "气体名称", "单位", "报警状态"}

    for j in range(df_data.shape[1]):
        col = _to_numeric_col(df_data.iloc[:, j])
        n_valid = int(col.notna().sum())
        if n_valid < len(df_data) * broken_ratio:
            continue

        vals = col.dropna().values
        if len(vals) > 0 and np.all(vals == broken_val):
            continue

        raw_name = _safe_name(row_name.iloc[j] if j < len(row_name) else "")
        fallback_name = ""
        if name_template_by_col is not None and j < len(name_template_by_col):
            fallback_name = _safe_name(name_template_by_col[j])
        name_str = raw_name if raw_name else fallback_name
        if not name_str:
            name_str = f"col_{j}"

        if name_str in skip_exact or name_str == "通道":
            continue

        valid_idx.append(j)
        cats.append(_safe_cat(row_cat.iloc[j] if j < len(row_cat) else ""))
        names.append(name_str)

    data = np.zeros((len(df_data), len(valid_idx)), dtype=np.float32)
    for out_j, src_j in enumerate(valid_idx):
        col = _to_numeric_col(df_data.iloc[:, src_j])
        vals = col.values.astype(np.float32)
        invalid = ~np.isfinite(vals) | (vals == broken_val)
        if preserve_missing:
            vals[invalid] = np.nan
        elif invalid.any():
            good = vals[~invalid]
            fill = float(np.median(good)) if len(good) > 0 else 0.0
            vals[invalid] = fill
        data[:, out_j] = vals

    if auto_reverse and len(data) > 0 and _should_reverse_experiment(data, cats, names):
        data = data[::-1].copy()

    return data, cats, names


def normalize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy normalization helper for already-imputed, fit-scope data."""
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds[stds < 1e-8] = 1.0
    return (data - means) / stds, means, stds


def fit_train_preprocessor(train_arrays: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Fit median imputation and z-score statistics using training data only."""
    if len(train_arrays) == 0:
        raise ValueError("Cannot fit preprocessing without training arrays.")

    n_sensors = int(train_arrays[0].shape[1])
    if n_sensors == 0 or any(arr.ndim != 2 or arr.shape[1] != n_sensors for arr in train_arrays):
        raise ValueError("Training arrays must be non-empty 2D arrays with aligned sensors.")

    train_data = _concat_arrays([np.asarray(arr, dtype=np.float32) for arr in train_arrays], axis=0)
    if len(train_data) == 0:
        raise ValueError("Cannot fit preprocessing on zero training rows.")

    fill_values = np.zeros(n_sensors, dtype=np.float32)
    for sensor_idx in range(n_sensors):
        values = train_data[:, sensor_idx]
        valid = values[np.isfinite(values)]
        if len(valid) > 0:
            fill_values[sensor_idx] = np.float32(np.median(valid))

    imputed_train = np.where(np.isfinite(train_data), train_data, fill_values[np.newaxis, :])
    means = imputed_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = imputed_train.std(axis=0, dtype=np.float64).astype(np.float32)
    stds[stds < 1e-8] = 1.0
    return {
        "fill_values": fill_values,
        "means": means,
        "stds": stds,
        "n_fit_rows": np.asarray([len(train_data)], dtype=np.int64),
    }


def apply_train_preprocessor(
    data: np.ndarray,
    preprocessor: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a training-fitted imputer and scaler to one array."""
    raw = np.asarray(data, dtype=np.float32)
    fill_values = np.asarray(preprocessor["fill_values"], dtype=np.float32)
    means = np.asarray(preprocessor["means"], dtype=np.float32)
    stds = np.asarray(preprocessor["stds"], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(fill_values):
        raise ValueError("Data shape does not match fitted preprocessing statistics.")

    imputed = np.where(np.isfinite(raw), raw, fill_values[np.newaxis, :]).astype(np.float32)
    normalized = ((imputed - means[np.newaxis, :]) / stds[np.newaxis, :]).astype(np.float32)
    return imputed, normalized


def generate_labels(
    data_raw: np.ndarray,
    cats: List[str],
    names: List[str],
    temp_thresh: float = 200.0,
    trans_thresh: float = 0.3,
    heat_thresh: float = 2.5,
    co_thresh: float = 100.0,
    min_modalities: int = 2,
    min_consecutive: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    t_steps, n_sensors = data_raw.shape
    sensor_types = infer_sensor_types(cats, names)
    label = np.zeros(t_steps, dtype=np.float32)
    groups = _group_sensor_indices(sensor_types)

    modality_flags: List[np.ndarray] = []
    if len(groups.get("temperature", [])) > 0:
        modality_flags.append(_family_series(data_raw, groups["temperature"], "temperature") > temp_thresh)
    if len(groups.get("transmittance", [])) > 0:
        modality_flags.append(_family_series(data_raw, groups["transmittance"], "transmittance") < trans_thresh)
    if len(groups.get("heat_flux", [])) > 0:
        modality_flags.append(_family_series(data_raw, groups["heat_flux"], "heat_flux") > heat_thresh)
    if len(groups.get("co", [])) > 0:
        modality_flags.append(_family_series(data_raw, groups["co"], "co") > co_thresh)

    if len(modality_flags) == 0:
        tte = np.full(t_steps, float(t_steps + 1), dtype=np.float32)
        return label, tte

    required_modalities = min(max(int(min_modalities), 1), len(modality_flags))
    min_consecutive = min(max(int(min_consecutive), 1), max(t_steps, 1))
    modality_hits = np.zeros(t_steps, dtype=np.int32)
    for flag in modality_flags:
        modality_hits += flag.astype(np.int32)

    danger = modality_hits >= required_modalities
    fire_onset = _first_sustained_true(danger, min_consecutive)
    if fire_onset is None:
        tte = np.full(t_steps, float(t_steps + 1), dtype=np.float32)
        return label, tte

    tte = np.zeros(t_steps, dtype=np.float32)
    label[fire_onset:] = 1.0
    if fire_onset > 0:
        tte[:fire_onset] = np.arange(fire_onset, 0, -1, dtype=np.float32)

    return label, tte


# ======================================================================
# dataset
# ======================================================================


class FireDataset(Dataset):
    def __init__(
        self,
        data_norm: np.ndarray,
        labels: np.ndarray,
        ttes: np.ndarray,
        window: int = 60,
        n_active: int = 20,
        training: bool = True,
        delta: int = 30,
    ):
        super().__init__()
        self.data = data_norm
        self.labels = labels
        self.ttes = ttes
        self.window = window
        self.n_sensors = data_norm.shape[1]
        self.n_active = min(n_active, self.n_sensors)
        self.training = training
        self.delta = delta

        t_steps = len(data_norm)
        self.indices = list(range(window - 1, t_steps))

        self.cls_labels = np.zeros(t_steps, dtype=np.float32)
        for t in range(t_steps):
            if ttes[t] <= delta:
                self.cls_labels[t] = 1.0

    def __len__(self):
        return len(self.indices)

    def _random_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_sensors, dtype=np.float32)
        chosen = np.random.choice(self.n_sensors, self.n_active, replace=False)
        mask[chosen] = 1.0
        return mask

    def __getitem__(self, idx: int):
        t_end = self.indices[idx]
        t_start = t_end - self.window + 1
        obs = self.data[t_start:t_end + 1].copy()

        if self.training:
            mask = self._random_mask()
        else:
            mask = np.ones(self.n_sensors, dtype=np.float32)

        obs_masked = obs * mask[np.newaxis, :]
        mask_expanded = np.tile(mask, (self.window, 1))
        x = np.concatenate([obs_masked, mask_expanded], axis=-1)

        label = self.cls_labels[t_end]
        tte = self.ttes[t_end]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(tte, dtype=torch.float32),
        )


# ======================================================================
# legacy prepare_data (kept for compatibility)
# ======================================================================


def prepare_data(cfg) -> Tuple[DataLoader, DataLoader, int]:
    """Compatibility wrapper using the leakage-safe train/validation protocol."""
    data_info = prepare_data_time_split(cfg)
    return data_info["train_loader"], data_info["val_loader"], data_info["n_sensors"]


# ======================================================================
# multi-experiment preparation
# ======================================================================


def _split_experiments_random(experiments: List[dict], cfg) -> Tuple[List[dict], List[dict], List[dict]]:
    n = len(experiments)
    if n < 3:
        raise ValueError("Need at least 3 experiment files for train/val/test split.")

    rng = np.random.default_rng(int(getattr(cfg, "split_seed", 42)))
    perm = rng.permutation(n)

    n_train = int(np.floor(n * float(cfg.train_ratio)))
    n_test = int(np.floor(n * float(cfg.test_ratio)))
    n_val = n - n_train - n_test

    if n_train < 1:
        n_train = 1
        n_val -= 1
    if n_test < 1:
        n_test = 1
        n_val -= 1
    while n_val < 1:
        if n_train > n_test and n_train > 1:
            n_train -= 1
            n_val += 1
        elif n_test > 1:
            n_test -= 1
            n_val += 1
        else:
            raise ValueError("Cannot satisfy non-empty train/val/test split.")

    train_idx = set(perm[:n_train].tolist())
    val_idx = set(perm[n_train:n_train + n_val].tolist())
    test_idx = set(perm[n_train + n_val:].tolist())

    # Preserve the original file order inside each split for stable timeline concatenation.
    train_exps = [exp for i, exp in enumerate(experiments) if i in train_idx]
    val_exps = [exp for i, exp in enumerate(experiments) if i in val_idx]
    test_exps = [exp for i, exp in enumerate(experiments) if i in test_idx]
    return train_exps, val_exps, test_exps


def _split_experiments_kfold(experiments: List[dict], cfg) -> Tuple[List[dict], List[dict], List[dict]]:
    n = len(experiments)
    n_splits = int(getattr(cfg, "experiment_kfold_n_splits", 5) or 5)
    if n_splits < 3:
        raise ValueError("experiment_kfold_n_splits must be at least 3 to keep train/val/test non-empty.")
    if n < n_splits:
        raise ValueError(f"Need at least {n_splits} experiment files for K-fold split, got {n}.")

    rng = np.random.default_rng(int(getattr(cfg, "split_seed", 42)))
    perm = rng.permutation(n)
    folds = [fold.astype(np.int64) for fold in np.array_split(perm, n_splits)]
    fold_idx = int(getattr(cfg, "experiment_kfold_index", 0) or 0) % n_splits
    val_idx = (fold_idx + 1) % n_splits

    test_set = set(folds[fold_idx].tolist())
    val_set = set(folds[val_idx].tolist())
    train_set = set()
    for i, fold in enumerate(folds):
        if i not in {fold_idx, val_idx}:
            train_set.update(fold.tolist())

    if not train_set or not val_set or not test_set:
        raise ValueError("K-fold split produced an empty train/val/test split.")

    train_exps = [exp for i, exp in enumerate(experiments) if i in train_set]
    val_exps = [exp for i, exp in enumerate(experiments) if i in val_set]
    test_exps = [exp for i, exp in enumerate(experiments) if i in test_set]
    return train_exps, val_exps, test_exps


def _build_concat_loader(experiments: List[dict], cfg, training: bool) -> DataLoader:
    ds_list: List[Dataset] = []
    for exp in experiments:
        ds_list.append(
            FireDataset(
                exp["data_norm"],
                exp["labels"],
                exp["ttes"],
                window=cfg.window_size,
                n_active=cfg.n_active,
                training=training,
                delta=cfg.delta,
            )
        )
    if len(ds_list) == 0:
        raise ValueError("Empty split, no datasets to build DataLoader.")
    ds: Dataset
    if len(ds_list) == 1:
        ds = ds_list[0]
    else:
        ds = ConcatDataset(ds_list)

    return _build_loader(ds, cfg, training)


def _align_multi_experiments(cfg, files: List[Path]) -> List[dict]:
    if len(files) == 0:
        return []

    # Build full-column fallback template from the first experiment.
    _, first_row_names, first_n_cols = _read_raw_header_rows(str(files[0]))
    template_names: List[str] = []
    for j in range(first_n_cols):
        nm = _safe_name(first_row_names[j] if j < len(first_row_names) else "")
        template_names.append(nm if nm else f"col_{j}")

    raw_exps: List[dict] = []
    for fp in files:
        data_raw, cats, names = load_and_clean(
            str(fp),
            broken_val=-200.0,
            broken_ratio=0.5,
            name_template_by_col=template_names,
            auto_reverse=getattr(cfg, "auto_reverse_experiments", False),
            preserve_missing=True,
        )
        keys = _channel_keys(names)
        raw_exps.append(
            {
                "exp_id": fp.stem,
                "file_path": str(fp),
                "data_raw": data_raw,
                "cats": cats,
                "names": names,
                "keys": keys,
            }
        )

    # Ordered common channels by first experiment's channel keys.
    first_keys = raw_exps[0]["keys"]
    key_sets = [set(exp["keys"]) for exp in raw_exps]
    common_key_set = set(first_keys)
    for s in key_sets[1:]:
        common_key_set &= s
    ordered_common_keys = [k for k in first_keys if k in common_key_set]
    if len(ordered_common_keys) == 0:
        raise ValueError("No common aligned channels found across experiments.")

    key_to_name_first = {k: n for k, n in zip(raw_exps[0]["keys"], raw_exps[0]["names"])}
    key_to_cat_first = {k: c for k, c in zip(raw_exps[0]["keys"], raw_exps[0]["cats"])}
    aligned_names = [key_to_name_first[k] for k in ordered_common_keys]
    aligned_cats = [key_to_cat_first.get(k, "") for k in ordered_common_keys]
    aligned_sensor_types = infer_sensor_types(aligned_cats, aligned_names)

    aligned_exps: List[dict] = []
    for exp in raw_exps:
        key_to_idx = {k: i for i, k in enumerate(exp["keys"])}
        keep_idx = [key_to_idx[k] for k in ordered_common_keys]
        data_raw = exp["data_raw"][:, keep_idx]
        aligned_exps.append(
            {
                "exp_id": exp["exp_id"],
                "file_path": exp["file_path"],
                "data_raw": data_raw,
                "cats": aligned_cats,
                "names": aligned_names,
                "sensor_types": aligned_sensor_types,
                "n_sensors": data_raw.shape[1],
                "length": data_raw.shape[0],
            }
        )
    return aligned_exps


def _fit_and_apply_multi_preprocessor(
    cfg,
    train_exps: List[dict],
    val_exps: List[dict],
    test_exps: List[dict],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Fit once on training experiments, then transform all split members."""
    preprocessor = fit_train_preprocessor([exp["data_raw"] for exp in train_exps])
    for exp in train_exps + val_exps + test_exps:
        data_raw, data_norm = apply_train_preprocessor(exp["data_raw"], preprocessor)
        labels, ttes = generate_labels(
            data_raw,
            exp["cats"],
            exp["names"],
            temp_thresh=cfg.temp_threshold,
            trans_thresh=cfg.trans_threshold,
            heat_thresh=cfg.heat_threshold,
            co_thresh=cfg.co_threshold,
            min_modalities=cfg.hazard_min_modalities,
            min_consecutive=cfg.hazard_min_consecutive,
        )
        exp.update(
            {
                "data_raw": data_raw,
                "data_norm": data_norm,
                "labels": labels,
                "ttes": ttes,
                "cls_labels": (ttes <= cfg.delta).astype(np.float32),
                "fire_onset": int(ttes[0]) if ttes[0] < len(ttes) else len(ttes),
            }
        )

    metadata = {
        "imputation": "per-sensor training median",
        "normalization": "per-sensor training z-score",
        "fit_scope": "training experiments only",
        "n_fit_rows": int(preprocessor["n_fit_rows"][0]),
    }
    return metadata, preprocessor


def _prepare_multi_experiment_data(cfg, files: List[Path]) -> dict:
    experiments = _align_multi_experiments(cfg, files)
    if len(experiments) == 0:
        raise ValueError("No experiments available after alignment.")

    experiment_split_mode = str(getattr(cfg, "experiment_split_mode", "random")).lower()
    if experiment_split_mode == "kfold":
        train_exps, val_exps, test_exps = _split_experiments_kfold(experiments, cfg)
    else:
        experiment_split_mode = "random"
        train_exps, val_exps, test_exps = _split_experiments_random(experiments, cfg)
    n_sensors = int(experiments[0]["n_sensors"])
    preprocessing, preprocessor = _fit_and_apply_multi_preprocessor(
        cfg, train_exps, val_exps, test_exps
    )

    train_loader = _build_concat_loader(train_exps, cfg, training=True)
    val_loader = _build_concat_loader(val_exps, cfg, training=False)
    test_loader = _build_concat_loader(test_exps, cfg, training=False)

    all_exps = train_exps + val_exps + test_exps
    train_data_concat = _concat_arrays([e["data_norm"] for e in train_exps], axis=0)
    val_data_concat = _concat_arrays([e["data_norm"] for e in val_exps], axis=0)
    test_data_concat = _concat_arrays([e["data_norm"] for e in test_exps], axis=0)
    all_data_concat = _concat_arrays([e["data_norm"] for e in all_exps], axis=0)

    all_labels = _concat_arrays([e["labels"] for e in all_exps], axis=0).astype(np.float32)
    all_ttes = _concat_arrays([e["ttes"] for e in all_exps], axis=0).astype(np.float32)
    all_cls = _concat_arrays([e["cls_labels"] for e in all_exps], axis=0).astype(np.float32)
    all_data_raw = _concat_arrays([e["data_raw"] for e in all_exps], axis=0)

    train_len = len(train_data_concat)
    val_len = len(val_data_concat)
    test_len = len(test_data_concat)
    total_len = train_len + val_len + test_len
    fire_onset = _first_positive_idx(all_cls)
    eval_fire_onset = _first_positive_idx(_concat_arrays([e["cls_labels"] for e in test_exps], axis=0))

    n_pos_train = int(_concat_arrays([e["cls_labels"] for e in train_exps], axis=0).sum())
    n_neg_train = int(sum(e["length"] for e in train_exps) - n_pos_train)

    print(f"[Data] multi-experiment mode: {len(experiments)} files")
    if experiment_split_mode == "kfold":
        print(
            f"[Data] split mode: kfold "
            f"fold={int(getattr(cfg, 'experiment_kfold_index', 0)) % int(getattr(cfg, 'experiment_kfold_n_splits', 5))}/"
            f"{int(getattr(cfg, 'experiment_kfold_n_splits', 5))}"
        )
    else:
        print(f"[Data] split mode: random seed={int(getattr(cfg, 'split_seed', 42))}")
    print(f"[Data] split files -> train={len(train_exps)}, val={len(val_exps)}, test={len(test_exps)}")
    print(f"[Data] aligned sensors: {n_sensors}")
    print(f"[Data] train/val/test steps: {train_len}/{val_len}/{test_len}")
    print(f"[Data] train pos/neg: {n_pos_train}/{n_neg_train}")
    print(f"[Data] preprocessing: {preprocessing['fit_scope']}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_sensors": n_sensors,
        "data_raw": all_data_raw,
        "data_norm": all_data_concat,
        "labels": all_labels,
        "ttes": all_ttes,
        "cls_labels": all_cls,
        "cats": experiments[0]["cats"],
        "names": experiments[0]["names"],
        "sensor_types": experiments[0]["sensor_types"],
        "fire_onset": fire_onset,
        "eval_fire_onset": eval_fire_onset,
        "train_range": (0, train_len),
        "val_range": (train_len, train_len + val_len),
        "test_range": (train_len + val_len, total_len),
        "sensor_names": experiments[0]["names"],
        "train_data_norm": train_data_concat,
        "val_data_norm": val_data_concat,
        "test_data_norm": test_data_concat,
        "train_experiments": train_exps,
        "val_experiments": val_exps,
        "test_experiments": test_exps,
        "all_experiments": all_exps,
        "split_mode": f"experiment_{experiment_split_mode}",
        "experiment_split_mode": experiment_split_mode,
        "experiment_kfold_n_splits": int(getattr(cfg, "experiment_kfold_n_splits", 0) or 0),
        "experiment_kfold_index": int(getattr(cfg, "experiment_kfold_index", 0) or 0),
        "preprocessing": preprocessing,
        "preprocessor": preprocessor,
    }


# ======================================================================
# public prepare_data_time_split
# ======================================================================


def prepare_data_time_split(cfg):
    """
    Prepare train/val/test data.

    Priority:
    1) Multi-experiment directory mode (converted_excel/*.xlsx by default).
    2) Legacy npy mode.
    3) Legacy single-file xlsx mode with time split.
    """
    data_dir = getattr(cfg, "data_dir", "")
    pattern = getattr(cfg, "experiment_glob", "*.xlsx")
    resolved_data_dir = _resolve_experiment_dir(data_dir)
    exp_files = _list_experiment_files(str(resolved_data_dir), pattern) if resolved_data_dir else []
    max_files = int(getattr(cfg, "experiment_max_files", 0) or 0)
    if max_files > 0 and len(exp_files) > max_files:
        exp_files = exp_files[:max_files]
        print(f"[Data] limit experiment files to first {len(exp_files)}")
    if len(exp_files) > 0:
        return _prepare_multi_experiment_data(cfg, exp_files)

    data_dir_path = resolved_data_dir
    if data_dir_path is not None and data_dir_path.exists() and data_dir_path.is_dir():
        msg = (
            f"[Data] no experiment files matched {pattern!r} under {data_dir_path}. "
            f"Falling back to legacy path: {cfg.data_path}"
        )
        if not Path(cfg.data_path).exists():
            raise FileNotFoundError(
                f"{msg}. Legacy file also not found."
            )
        print(msg)
    elif data_dir:
        legacy_exists = Path(cfg.data_path).exists()
        data_dir_legacy = os.path.dirname(cfg.data_path) if hasattr(cfg, "data_path") else "."
        npy_exists = has_npy_data(data_dir_legacy)
        if not legacy_exists and not npy_exists:
            raise FileNotFoundError(
                f"[Data] experiment directory not found: {data_dir}. "
                f"Also no legacy data found at {cfg.data_path}."
            )

    data_dir_legacy = os.path.dirname(cfg.data_path) if hasattr(cfg, "data_path") else "."
    if has_npy_data(data_dir_legacy):
        print("[Data] found npy files, using npy loader")
        warnings.warn(
            "Using preprocessed .npy arrays. Their train-only preprocessing cannot "
            "be verified or refit because the raw values are unavailable.",
            RuntimeWarning,
        )
        npy_info = load_from_npy(data_dir_legacy)
        data_norm = npy_info["data_norm"]
        data_raw = data_norm.copy()
        n_sensors = npy_info["n_sensors"]
        t_steps = data_norm.shape[0]
        labels = npy_info["labels"]
        ttes = npy_info["ttes"]
        cls_labels = npy_info["cls_labels"]
        fire_onset = npy_info["fire_onset"]
        cats = [f"cat_{i}" for i in range(n_sensors)]
        names = [f"sensor_{i}" for i in range(n_sensors)]
        preprocessing = {
            "imputation": "external preprocessed input",
            "normalization": "external preprocessed input",
            "fit_scope": "unknown; raw data unavailable",
        }
        preprocessor = None
    else:
        data_raw, cats, names = load_and_clean(
            cfg.data_path,
            auto_reverse=getattr(cfg, "auto_reverse_experiments", False),
            preserve_missing=True,
        )
        n_sensors = data_raw.shape[1]
        t_steps = data_raw.shape[0]
        train_end_for_preprocessing = int(t_steps * cfg.train_ratio)
        preprocessor = fit_train_preprocessor([data_raw[:train_end_for_preprocessing]])
        data_raw, data_norm = apply_train_preprocessor(data_raw, preprocessor)
        preprocessing = {
            "imputation": "per-sensor training median",
            "normalization": "per-sensor training z-score",
            "fit_scope": "training time range only",
            "n_fit_rows": int(preprocessor["n_fit_rows"][0]),
        }
        labels, ttes = generate_labels(
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
        fire_onset = int(ttes[0]) if ttes[0] < t_steps else t_steps
        cls_labels = np.zeros(t_steps, dtype=np.float32)
        for t in range(t_steps):
            if ttes[t] <= cfg.delta:
                cls_labels[t] = 1.0

    train_end = int(t_steps * cfg.train_ratio)
    val_end = int(t_steps * (1.0 - cfg.test_ratio))
    w = cfg.window_size
    train_range = (0, train_end)
    val_range = (train_end, val_end)
    test_range = (val_end, t_steps)
    print(f"[Data] time split: train={train_range}, val={val_range}, test={test_range}")

    train_ds = FireDataset(
        data_norm[:train_end],
        labels[:train_end],
        ttes[:train_end],
        window=w,
        n_active=cfg.n_active,
        training=True,
        delta=cfg.delta,
    )
    val_ds = FireDataset(
        data_norm[max(0, train_end - w + 1):val_end],
        labels[max(0, train_end - w + 1):val_end],
        ttes[max(0, train_end - w + 1):val_end],
        window=w,
        n_active=cfg.n_active,
        training=False,
        delta=cfg.delta,
    )
    test_ds = FireDataset(
        data_norm[max(0, val_end - w + 1):t_steps],
        labels[max(0, val_end - w + 1):t_steps],
        ttes[max(0, val_end - w + 1):t_steps],
        window=w,
        n_active=cfg.n_active,
        training=False,
        delta=cfg.delta,
    )

    train_loader = _build_loader(train_ds, cfg, training=True)
    val_loader = _build_loader(val_ds, cfg, training=False)
    test_loader = _build_loader(test_ds, cfg, training=False)

    n_pos_train = int(cls_labels[:train_end].sum())
    n_neg_train = int(train_end - n_pos_train)
    print(f"[Data] train/val/test samples: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"[Data] train pos/neg: {n_pos_train}/{n_neg_train}")
    sensor_types = infer_sensor_types(cats, names)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_sensors": n_sensors,
        "data_raw": data_raw,
        "data_norm": data_norm,
        "labels": labels,
        "ttes": ttes,
        "cls_labels": cls_labels,
        "cats": cats,
        "names": names,
        "sensor_types": sensor_types,
        "fire_onset": fire_onset,
        "eval_fire_onset": _first_positive_idx(cls_labels[max(0, val_end - w + 1):]),
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
        "sensor_names": names,
        "train_data_norm": data_norm[:train_end],
        "val_data_norm": data_norm[max(0, train_end - w + 1):val_end],
        "test_data_norm": data_norm[max(0, val_end - w + 1):t_steps],
        "train_experiments": [],
        "val_experiments": [],
        "test_experiments": [],
        "all_experiments": [],
        "split_mode": "time_single",
        "preprocessing": preprocessing,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from config import Config

    cfg = Config()
    data_info = prepare_data_time_split(cfg)
    print(f"split_mode={data_info.get('split_mode')}, n_sensors={data_info['n_sensors']}")
