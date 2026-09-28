"""
compare_predictors.py

Compare predictor backbones for the risk prediction task under the same
partial-observation budget:
  1. LSTM baseline   (reuse project MaskAwareRiskPredictor)
  2. 1D CNN
  3. Transformer
  4. TCN
  5. PatchTST-like Transformer

The script prefers project integrations when available, but still works with
local fallback data loading from:
  - full_data.npy
  - y_labels.npy
  - tte_labels.npy

If the numpy files are not available, it falls back to the existing Excel-based
pipeline in dataset.py.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1")


def _configure_windows_dll_search_paths() -> None:
    """Improve DLL resolution for pip-installed PyTorch inside conda on Windows."""
    if os.name != "nt":
        return

    prefix = Path(sys.exec_prefix)
    candidate_dirs = [
        prefix / "Library" / "bin",
        prefix / "Lib" / "site-packages" / "torch" / "lib",
        prefix / "bin",
    ]

    existing_dirs = [str(path) for path in candidate_dirs if path.exists()]
    for dll_dir in existing_dirs:
        try:
            os.add_dll_directory(dll_dir)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    missing_dirs = [dll_dir for dll_dir in existing_dirs if dll_dir not in path_parts]
    if missing_dirs:
        os.environ["PATH"] = os.pathsep.join(missing_dirs + path_parts)


_configure_windows_dll_search_paths()

def _import_torch_with_retry():
    """Retry torch import once after forcing Windows DLL and OpenMP settings."""
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        return _torch, _nn, _F
    except OSError as first_error:
        if os.name != "nt":
            raise

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"
        _configure_windows_dll_search_paths()

        sys.modules.pop("torch", None)
        try:
            import torch as _torch
            import torch.nn as _nn
            import torch.nn.functional as _F
            return _torch, _nn, _F
        except OSError as second_error:
            raise OSError(
                f"{second_error}. Windows PyTorch import failed after retry. "
                "This usually indicates a DLL/OpenMP conflict in the active conda environment."
            ) from first_error


torch, nn, F = _import_torch_with_retry()
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import MaskAwareRiskPredictor, MultiTaskLoss

try:
    from config import Config as ProjectConfig
except ImportError:
    ProjectConfig = None

try:
    from dataset import generate_labels, load_and_clean, normalize
except ImportError:
    generate_labels = None
    load_and_clean = None
    normalize = None

try:
    from dataset import prepare_data_time_split
    from dataset import _window_labels_from_dataset as project_window_labels_from_dataset
except ImportError:
    prepare_data_time_split = None
    project_window_labels_from_dataset = None


@dataclass
class CompareConfig:
    """Configuration for the architecture comparison experiment."""

    data_protocol: str = "project"
    data_dir: str = str(ROOT_DIR)
    data_path: Optional[str] = None
    save_dir: str = str(ROOT_DIR / "checkpoints" / "compare_predictors")
    experiment_glob: str = "*.xlsx"
    experiment_max_files: int = 0
    experiment_split_mode: str = "random"
    experiment_kfold_n_splits: int = 5
    experiment_kfold_index: int = 0

    window_size: int = 60
    delta: int = 30
    mask_budget: int = 10

    batch_size: int = 64
    epochs: int = 50
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lambda_tte: float = 0.5
    huber_delta: float = 1.0

    hidden_dim: int = 128
    num_layers: int = 2
    lstm_dropout: float = 0.2
    head_dropout: float = 0.3

    cnn_channels: Tuple[int, int, int] = (64, 128, 256)
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_ff_dim: int = 1024
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    tcn_channels: Tuple[int, ...] = (128, 128, 128, 128)
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.15
    patch_len: int = 10
    patch_stride: int = 5
    patch_d_model: int = 128
    patch_nhead: int = 8
    patch_ff_dim: int = 256
    patch_layers: int = 2
    patch_dropout: float = 0.1

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_mode: str = "auto"

    temp_threshold: float = 200.0
    trans_threshold: float = 0.3
    heat_threshold: float = 2.5
    co_threshold: float = 100.0
    hazard_min_modalities: int = 2
    hazard_min_consecutive: int = 3
    auto_reverse_experiments: bool = True

    device: str = "cuda"
    seed: int = 42
    seeds: Tuple[int, ...] = (42,)
    num_workers: int = 0

    model_names: Tuple[str, ...] = ("lstm", "cnn", "transformer", "tcn", "patchtst")
    alarm_threshold: float = 0.5
    min_pos_weight: float = 0.1
    max_pos_weight: float = 30.0
    print_every: int = 1

    extra: Dict[str, Any] = field(default_factory=dict)


def build_default_config() -> CompareConfig:
    """Construct defaults, borrowing project-level values when available."""
    cfg = CompareConfig()
    if ProjectConfig is not None:
        project_cfg = ProjectConfig()
        cfg.data_path = getattr(project_cfg, "data_path", None)
        cfg.data_dir = getattr(project_cfg, "data_dir", cfg.data_dir)
        cfg.experiment_glob = getattr(project_cfg, "experiment_glob", cfg.experiment_glob)
        cfg.experiment_max_files = getattr(project_cfg, "experiment_max_files", cfg.experiment_max_files)
        cfg.experiment_split_mode = getattr(project_cfg, "experiment_split_mode", cfg.experiment_split_mode)
        cfg.experiment_kfold_n_splits = getattr(project_cfg, "experiment_kfold_n_splits", cfg.experiment_kfold_n_splits)
        cfg.experiment_kfold_index = getattr(project_cfg, "experiment_kfold_index", cfg.experiment_kfold_index)
        cfg.window_size = getattr(project_cfg, "window_size", cfg.window_size)
        cfg.delta = getattr(project_cfg, "delta", cfg.delta)
        cfg.mask_budget = getattr(project_cfg, "n_active", cfg.mask_budget)
        cfg.batch_size = getattr(project_cfg, "batch_size", cfg.batch_size)
        cfg.lr = getattr(project_cfg, "lr", cfg.lr)
        cfg.weight_decay = getattr(project_cfg, "weight_decay", cfg.weight_decay)
        cfg.lambda_tte = getattr(project_cfg, "lambda_tte", cfg.lambda_tte)
        cfg.huber_delta = getattr(project_cfg, "huber_delta", cfg.huber_delta)
        cfg.patience = getattr(project_cfg, "patience", cfg.patience)
        cfg.train_ratio = getattr(project_cfg, "train_ratio", cfg.train_ratio)
        cfg.test_ratio = getattr(project_cfg, "test_ratio", cfg.test_ratio)
        cfg.val_ratio = 1.0 - cfg.train_ratio - cfg.test_ratio
        cfg.temp_threshold = getattr(project_cfg, "temp_threshold", cfg.temp_threshold)
        cfg.trans_threshold = getattr(project_cfg, "trans_threshold", cfg.trans_threshold)
        cfg.heat_threshold = getattr(project_cfg, "heat_threshold", cfg.heat_threshold)
        cfg.co_threshold = getattr(project_cfg, "co_threshold", cfg.co_threshold)
        cfg.hazard_min_modalities = getattr(project_cfg, "hazard_min_modalities", cfg.hazard_min_modalities)
        cfg.hazard_min_consecutive = getattr(project_cfg, "hazard_min_consecutive", cfg.hazard_min_consecutive)
        cfg.auto_reverse_experiments = getattr(project_cfg, "auto_reverse_experiments", cfg.auto_reverse_experiments)
        cfg.device = getattr(project_cfg, "device", cfg.device)
        cfg.hidden_dim = getattr(project_cfg, "hidden_dim", cfg.hidden_dim)
        cfg.num_layers = getattr(project_cfg, "num_layers", cfg.num_layers)
        cfg.lstm_dropout = getattr(project_cfg, "lstm_dropout", cfg.lstm_dropout)
        cfg.head_dropout = getattr(project_cfg, "head_dropout", cfg.head_dropout)
        cfg.min_pos_weight = getattr(project_cfg, "min_pos_weight", cfg.min_pos_weight)
        cfg.max_pos_weight = getattr(project_cfg, "max_pos_weight", cfg.max_pos_weight)
    return cfg


def parse_args() -> CompareConfig:
    default_cfg = build_default_config()

    parser = argparse.ArgumentParser(description="Compare LSTM/CNN/Transformer/TCN/PatchTST predictors")
    parser.add_argument(
        "--data-protocol",
        type=str,
        default=default_cfg.data_protocol,
        choices=["project", "legacy"],
        help="project uses dataset.prepare_data_time_split; legacy uses this script's old array loader.",
    )
    parser.add_argument("--data-dir", type=str, default=default_cfg.data_dir)
    parser.add_argument("--data-path", type=str, default=default_cfg.data_path)
    parser.add_argument("--save-dir", type=str, default=default_cfg.save_dir)
    parser.add_argument("--experiment-glob", type=str, default=default_cfg.experiment_glob)
    parser.add_argument("--experiment-max-files", type=int, default=default_cfg.experiment_max_files)
    parser.add_argument("--experiment-split-mode", type=str, default=default_cfg.experiment_split_mode, choices=["random", "kfold"])
    parser.add_argument("--experiment-kfold-n-splits", type=int, default=default_cfg.experiment_kfold_n_splits)
    parser.add_argument("--experiment-kfold-index", type=int, default=default_cfg.experiment_kfold_index)
    parser.add_argument("--window-size", type=int, default=default_cfg.window_size)
    parser.add_argument("--delta", type=int, default=default_cfg.delta)
    parser.add_argument("--mask-budget", type=int, default=default_cfg.mask_budget)
    parser.add_argument("--batch-size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=default_cfg.epochs)
    parser.add_argument("--patience", type=int, default=default_cfg.patience)
    parser.add_argument("--lr", type=float, default=default_cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=default_cfg.weight_decay)
    parser.add_argument("--device", type=str, default=default_cfg.device)
    parser.add_argument("--seed", type=int, default=default_cfg.seed)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in default_cfg.seeds),
        help="Comma-separated seeds for repeated runs, e.g. 42,43,44.",
    )
    parser.add_argument("--train-ratio", type=float, default=default_cfg.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=default_cfg.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=default_cfg.test_ratio)
    parser.add_argument("--split-mode", type=str, default=default_cfg.split_mode, choices=["auto", "time", "stratified"])
    parser.add_argument("--num-workers", type=int, default=default_cfg.num_workers)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(default_cfg.model_names),
        help="Comma-separated model names: lstm,cnn,transformer,tcn,patchtst.",
    )
    parser.add_argument("--transformer-d-model", type=int, default=default_cfg.transformer_d_model)
    parser.add_argument("--transformer-nhead", type=int, default=default_cfg.transformer_nhead)
    parser.add_argument("--transformer-ff-dim", type=int, default=default_cfg.transformer_ff_dim)
    parser.add_argument("--transformer-layers", type=int, default=default_cfg.transformer_layers)
    parser.add_argument("--transformer-dropout", type=float, default=default_cfg.transformer_dropout)
    parser.add_argument("--patch-len", type=int, default=default_cfg.patch_len)
    parser.add_argument("--patch-stride", type=int, default=default_cfg.patch_stride)
    parser.add_argument("--patch-d-model", type=int, default=default_cfg.patch_d_model)
    parser.add_argument("--patch-layers", type=int, default=default_cfg.patch_layers)
    parser.add_argument("--tcn-kernel-size", type=int, default=default_cfg.tcn_kernel_size)
    args = parser.parse_args()

    cfg = build_default_config()
    cfg.data_protocol = args.data_protocol
    cfg.data_dir = args.data_dir
    cfg.data_path = args.data_path
    cfg.save_dir = args.save_dir
    cfg.experiment_glob = args.experiment_glob
    cfg.experiment_max_files = args.experiment_max_files
    cfg.experiment_split_mode = args.experiment_split_mode
    cfg.experiment_kfold_n_splits = args.experiment_kfold_n_splits
    cfg.experiment_kfold_index = args.experiment_kfold_index
    cfg.window_size = args.window_size
    cfg.delta = args.delta
    cfg.mask_budget = args.mask_budget
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.patience = args.patience
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.seeds = parse_seed_list(args.seeds, fallback=args.seed)
    cfg.train_ratio = args.train_ratio
    cfg.val_ratio = args.val_ratio
    cfg.test_ratio = args.test_ratio
    cfg.split_mode = args.split_mode
    cfg.num_workers = args.num_workers
    cfg.model_names = tuple(
        item.strip().lower()
        for item in str(args.models).split(",")
        if item.strip()
    )
    cfg.transformer_d_model = args.transformer_d_model
    cfg.transformer_nhead = args.transformer_nhead
    cfg.transformer_ff_dim = args.transformer_ff_dim
    cfg.transformer_layers = args.transformer_layers
    cfg.transformer_dropout = args.transformer_dropout
    cfg.patch_len = args.patch_len
    cfg.patch_stride = args.patch_stride
    cfg.patch_d_model = args.patch_d_model
    cfg.patch_layers = args.patch_layers
    cfg.tcn_kernel_size = args.tcn_kernel_size

    total_ratio = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {total_ratio:.4f}"
        )
    return cfg


def parse_seed_list(text: str, fallback: int = 42) -> Tuple[int, ...]:
    """Parse a comma-separated seed list while keeping CLI errors explicit."""
    if text is None or str(text).strip() == "":
        return (int(fallback),)
    seeds: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            seeds.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid seed value: {item!r}") from exc
    return tuple(seeds) if seeds else (int(fallback),)


def set_seed(seed: int) -> None:
    """Make model initialization, shuffling and mask sampling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Resolve the requested device with a safe CPU fallback."""
    if device_name == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def generate_sensor_mask(n_sensors: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Sample exactly m active sensors without replacement."""
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")
    m = min(max(int(m), 1), n_sensors)
    mask = np.zeros(n_sensors, dtype=np.float32)
    chosen = rng.choice(n_sensors, size=m, replace=False)
    mask[chosen] = 1.0
    return mask


class WindowedRiskDataset(Dataset):
    """
    Sliding-window dataset that always returns the unified input shape (W, 2N).

    Train split:
      - random mask sampled on each access
    Validation/Test split:
      - deterministic per-sample mask generated from seed + idx
    """

    def __init__(
        self,
        data: np.ndarray,
        cls_labels: np.ndarray,
        tte_labels: np.ndarray,
        window_size: int,
        mask_budget: int,
        training: bool,
        seed: int,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError(f"data must be 2D (T, N), got shape={data.shape}")
        if len(data) != len(cls_labels) or len(data) != len(tte_labels):
            raise ValueError("data, cls_labels and tte_labels must have matching length")
        if len(data) < window_size:
            raise ValueError(
                f"segment length {len(data)} is smaller than window_size={window_size}"
            )

        self.data = data.astype(np.float32)
        self.cls_labels = cls_labels.astype(np.float32)
        self.tte_labels = tte_labels.astype(np.float32)
        self.window_size = int(window_size)
        self.mask_budget = int(mask_budget)
        self.training = bool(training)
        self.seed = int(seed)
        self.n_sensors = self.data.shape[1]
        if indices is None:
            self.indices = np.arange(self.window_size - 1, len(self.data), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
        self._train_rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _build_mask(self, idx: int) -> np.ndarray:
        if self.training:
            return generate_sensor_mask(self.n_sensors, self.mask_budget, self._train_rng)
        eval_rng = np.random.default_rng(self.seed + int(idx))
        return generate_sensor_mask(self.n_sensors, self.mask_budget, eval_rng)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_end = int(self.indices[idx])
        t_start = t_end - self.window_size + 1
        obs = self.data[t_start : t_end + 1]
        mask = self._build_mask(idx)
        mask_window = np.repeat(mask[None, :], self.window_size, axis=0)
        x = np.concatenate([obs * mask_window, mask_window], axis=1)

        label = self.cls_labels[t_end]
        tte = self.tte_labels[t_end]
        return (
            torch.from_numpy(x),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(tte, dtype=torch.float32),
        )


def _project_cfg_from_compare(cfg: CompareConfig, seed: Optional[int] = None):
    if ProjectConfig is None:
        raise RuntimeError("Project Config is unavailable; use --data-protocol legacy.")
    project_cfg = ProjectConfig()
    project_cfg.data_path = cfg.data_path or getattr(project_cfg, "data_path", "")
    project_cfg.data_dir = cfg.data_dir
    project_cfg.experiment_glob = cfg.experiment_glob
    project_cfg.experiment_max_files = int(cfg.experiment_max_files)
    project_cfg.experiment_split_mode = str(cfg.experiment_split_mode)
    project_cfg.experiment_kfold_n_splits = int(cfg.experiment_kfold_n_splits)
    project_cfg.experiment_kfold_index = int(cfg.experiment_kfold_index)
    project_cfg.window_size = int(cfg.window_size)
    project_cfg.delta = int(cfg.delta)
    project_cfg.train_ratio = float(cfg.train_ratio)
    project_cfg.val_ratio = float(cfg.val_ratio)
    project_cfg.test_ratio = float(cfg.test_ratio)
    project_cfg.split_seed = int(cfg.seed if seed is None else seed)
    project_cfg.n_active = int(cfg.mask_budget)
    project_cfg.batch_size = int(cfg.batch_size)
    project_cfg.data_num_workers = int(cfg.num_workers)
    project_cfg.temp_threshold = float(cfg.temp_threshold)
    project_cfg.trans_threshold = float(cfg.trans_threshold)
    project_cfg.heat_threshold = float(cfg.heat_threshold)
    project_cfg.co_threshold = float(cfg.co_threshold)
    project_cfg.hazard_min_modalities = int(cfg.hazard_min_modalities)
    project_cfg.hazard_min_consecutive = int(cfg.hazard_min_consecutive)
    project_cfg.auto_reverse_experiments = bool(cfg.auto_reverse_experiments)
    project_cfg.device = cfg.device
    return project_cfg


def _label_counts_from_dataset(ds: Dataset) -> Dict[float, int]:
    if project_window_labels_from_dataset is None:
        return {}
    try:
        labels = project_window_labels_from_dataset(ds)
    except Exception:
        return {}
    if len(labels) == 0:
        return {}
    uniq, cnt = np.unique(labels.astype(np.float32), return_counts=True)
    return {float(k): int(v) for k, v in zip(uniq, cnt)}


def _project_split_dataset(
    experiments: List[Dict[str, Any]],
    cfg: CompareConfig,
    training: bool,
    seed: int,
) -> Dataset:
    datasets: List[Dataset] = []
    for exp_idx, exp in enumerate(experiments):
        if len(exp["data_norm"]) < cfg.window_size:
            continue
        datasets.append(
            WindowedRiskDataset(
                exp["data_norm"],
                exp["cls_labels"],
                exp["ttes"],
                window_size=cfg.window_size,
                mask_budget=cfg.mask_budget,
                training=training,
                seed=seed + exp_idx * 997,
            )
        )
    if not datasets:
        raise ValueError("No experiment is long enough for the requested window size.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_project_dataloaders(
    cfg: CompareConfig,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Reuse the project's preprocessing and experiment-level split protocol."""
    if prepare_data_time_split is None:
        raise RuntimeError("prepare_data_time_split is unavailable; use --data-protocol legacy.")
    seed = int(cfg.seed if seed is None else seed)
    project_cfg = _project_cfg_from_compare(cfg, seed=seed)
    data_info = prepare_data_time_split(project_cfg)

    split_mode = data_info.get("split_mode", "time_single")
    if split_mode.startswith("experiment_") and data_info.get("train_experiments"):
        train_dataset = _project_split_dataset(data_info["train_experiments"], cfg, training=True, seed=seed)
        val_dataset = _project_split_dataset(data_info["val_experiments"], cfg, training=False, seed=seed + 10_000)
        test_dataset = _project_split_dataset(data_info["test_experiments"], cfg, training=False, seed=seed + 20_000)
    else:
        train_range = data_info["train_range"]
        val_range = data_info["val_range"]
        test_range = data_info["test_range"]
        data = data_info["data_norm"]
        cls = data_info["cls_labels"]
        ttes = data_info["ttes"]
        w = cfg.window_size

        train_dataset = WindowedRiskDataset(
            data[train_range[0]:train_range[1]],
            cls[train_range[0]:train_range[1]],
            ttes[train_range[0]:train_range[1]],
            window_size=w,
            mask_budget=cfg.mask_budget,
            training=True,
            seed=seed,
        )
        val_start = max(0, int(val_range[0]) - w + 1)
        test_start = max(0, int(test_range[0]) - w + 1)
        val_dataset = WindowedRiskDataset(
            data[val_start:val_range[1]],
            cls[val_start:val_range[1]],
            ttes[val_start:val_range[1]],
            window_size=w,
            mask_budget=cfg.mask_budget,
            training=False,
            seed=seed + 10_000,
        )
        test_dataset = WindowedRiskDataset(
            data[test_start:test_range[1]],
            cls[test_start:test_range[1]],
            ttes[test_start:test_range[1]],
            window_size=w,
            mask_budget=cfg.mask_budget,
            training=False,
            seed=seed + 20_000,
        )

    loader_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    train_counts = _label_counts_from_dataset(train_dataset)
    val_counts = _label_counts_from_dataset(val_dataset)
    test_counts = _label_counts_from_dataset(test_dataset)
    train_positive = int(train_counts.get(1.0, 0))
    train_negative = int(train_counts.get(0.0, 0))
    meta = {
        "source": "project_prepare_data_time_split",
        "n_sensors": int(data_info["n_sensors"]),
        "input_dim": int(2 * data_info["n_sensors"]),
        "n_timesteps": int(len(data_info["data_norm"])),
        "train_range": data_info.get("train_range"),
        "val_range": data_info.get("val_range"),
        "test_range": data_info.get("test_range"),
        "train_positive": train_positive,
        "train_negative": train_negative,
        "min_pos_weight": float(cfg.min_pos_weight),
        "max_pos_weight": float(cfg.max_pos_weight),
        "mask_budget": int(cfg.mask_budget),
        "split_mode": split_mode,
        "train_label_counts": train_counts,
        "val_label_counts": val_counts,
        "test_label_counts": test_counts,
        "n_train_experiments": len(data_info.get("train_experiments", [])),
        "n_val_experiments": len(data_info.get("val_experiments", [])),
        "n_test_experiments": len(data_info.get("test_experiments", [])),
        "train_experiment_ids": [str(exp.get("exp_id", "")) for exp in data_info.get("train_experiments", [])],
        "val_experiment_ids": [str(exp.get("exp_id", "")) for exp in data_info.get("val_experiments", [])],
        "test_experiment_ids": [str(exp.get("exp_id", "")) for exp in data_info.get("test_experiments", [])],
    }
    return train_loader, val_loader, test_loader, meta


class LSTMBaseline(nn.Module):
    """Thin wrapper around the existing project LSTM predictor."""

    def __init__(self, n_sensors: int, cfg: CompareConfig) -> None:
        super().__init__()
        self.backbone = MaskAwareRiskPredictor(
            n_sensors=n_sensors,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            lstm_dropout=cfg.lstm_dropout,
            head_dropout=cfg.head_dropout,
        )
        self.feature_dim = cfg.hidden_dim
        self.cls_head = self.backbone.cls_head
        self.tte_head = self.backbone.tte_head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(x)

    def encode_belief(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_belief(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone.predict(x)


class CNN1DPredictor(nn.Module):
    """1D CNN predictor that internally transposes inputs to (B, 2N, W)."""

    def __init__(self, n_sensors: int, cfg: CompareConfig) -> None:
        super().__init__()
        input_channels = 2 * n_sensors
        c1, c2, c3 = cfg.cnn_channels

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_dim = c3
        self.cls_head = nn.Linear(c3, 1)
        self.tte_head = nn.Sequential(nn.Linear(c3, 1), nn.Softplus())
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode_belief(x)
        logit = self.cls_head(feat).squeeze(-1)
        tte_hat = self.tte_head(feat).squeeze(-1)
        return logit, tte_hat

    def encode_belief(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.transpose(1, 2)).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logit, tte_hat = self.forward(x)
        return torch.sigmoid(logit), tte_hat


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for a fixed maximum sequence length."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerPredictor(nn.Module):
    """Transformer encoder with mean pooling over time."""

    def __init__(self, n_sensors: int, cfg: CompareConfig) -> None:
        super().__init__()
        input_dim = 2 * n_sensors
        d_model = cfg.transformer_d_model
        if d_model <= 0 or cfg.transformer_ff_dim <= 0 or cfg.transformer_layers <= 0:
            raise ValueError("Transformer dimensions and layer count must be positive")
        if cfg.transformer_nhead <= 0 or d_model % cfg.transformer_nhead != 0:
            raise ValueError(
                "transformer_d_model must be divisible by transformer_nhead, got "
                f"{d_model} and {cfg.transformer_nhead}"
            )

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=cfg.window_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.transformer_nhead,
            dim_feedforward=cfg.transformer_ff_dim,
            dropout=cfg.transformer_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, 1)
        self.tte_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.feature_dim = d_model
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode_belief(x)
        logit = self.cls_head(feat).squeeze(-1)
        tte_hat = self.tte_head(feat).squeeze(-1)
        return logit, tte_hat

    def encode_belief(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        return self.norm(h.mean(dim=1))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logit, tte_hat = self.forward(x)
        return torch.sigmoid(logit), tte_hat


class Chomp1d(nn.Module):
    """Remove right padding so dilated Conv1d stays causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual TCN block with two dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TCNPredictor(nn.Module):
    """Temporal Convolutional Network baseline with dilated residual blocks."""

    def __init__(self, n_sensors: int, cfg: CompareConfig) -> None:
        super().__init__()
        input_channels = 2 * n_sensors
        channels = tuple(int(c) for c in cfg.tcn_channels)
        if len(channels) == 0:
            channels = (cfg.hidden_dim, cfg.hidden_dim)

        layers: List[nn.Module] = []
        in_channels = input_channels
        kernel_size = max(2, int(cfg.tcn_kernel_size))
        for layer_idx, out_channels in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** layer_idx,
                    dropout=float(cfg.tcn_dropout),
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(channels[-1])
        self.feature_dim = channels[-1]
        self.cls_head = nn.Linear(channels[-1], 1)
        self.tte_head = nn.Sequential(nn.Linear(channels[-1], 1), nn.Softplus())
        self._init_heads()

    def _init_heads(self) -> None:
        for module in [self.cls_head, self.tte_head[0]]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode_belief(x)
        logit = self.cls_head(feat).squeeze(-1)
        tte_hat = self.tte_head(feat).squeeze(-1)
        return logit, tte_hat

    def encode_belief(self, x: torch.Tensor) -> torch.Tensor:
        h = self.network(x.transpose(1, 2))
        return self.norm(h[:, :, -1])

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logit, tte_hat = self.forward(x)
        return torch.sigmoid(logit), tte_hat


class PatchTSTPredictor(nn.Module):
    """
    PatchTST-like encoder for this task.

    It patches the time axis of the masked multivariate window, embeds each
    flattened patch, applies a Transformer encoder over patches, then uses mean
    pooling for classification and TTE regression.
    """

    def __init__(self, n_sensors: int, cfg: CompareConfig) -> None:
        super().__init__()
        input_dim = 2 * n_sensors
        patch_len = max(2, min(int(cfg.patch_len), int(cfg.window_size)))
        patch_stride = max(1, int(cfg.patch_stride))
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_patches = max(1, 1 + (int(cfg.window_size) - patch_len) // patch_stride)
        d_model = int(cfg.patch_d_model)

        self.patch_proj = nn.Linear(patch_len * input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.patch_nhead),
            dim_feedforward=int(cfg.patch_ff_dim),
            dropout=float(cfg.patch_dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(cfg.patch_layers))
        self.norm = nn.LayerNorm(d_model)
        self.feature_dim = d_model
        self.cls_head = nn.Linear(d_model, 1)
        self.tte_head = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode_belief(x)
        logit = self.cls_head(feat).squeeze(-1)
        tte_hat = self.tte_head(feat).squeeze(-1)
        return logit, tte_hat

    def encode_belief(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches[:, :self.n_patches]
        patches = patches.contiguous().view(patches.shape[0], patches.shape[1], -1)
        h = self.patch_proj(patches) + self.pos_embedding[:, : patches.shape[1]]
        h = self.encoder(h)
        return self.norm(h.mean(dim=1))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logit, tte_hat = self.forward(x)
        return torch.sigmoid(logit), tte_hat


def build_model(name: str, n_sensors: int, cfg: CompareConfig) -> nn.Module:
    """Factory for predictor backbones."""
    name = name.lower()
    if name == "lstm":
        return LSTMBaseline(n_sensors=n_sensors, cfg=cfg)
    if name == "cnn":
        return CNN1DPredictor(n_sensors=n_sensors, cfg=cfg)
    if name == "transformer":
        return TransformerPredictor(n_sensors=n_sensors, cfg=cfg)
    if name == "tcn":
        return TCNPredictor(n_sensors=n_sensors, cfg=cfg)
    if name in {"patchtst", "patch_tst", "patch"}:
        return PatchTSTPredictor(n_sensors=n_sensors, cfg=cfg)
    raise ValueError(f"Unsupported model name: {name}")


def _try_external_load_data(
    cfg: CompareConfig,
) -> Optional[Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]]:
    """Use a project-level load_data() if one exists and is compatible."""
    try:
        from dataset import load_data as external_load_data
    except ImportError:
        return None

    kwargs_map = {
        "data_dir": cfg.data_dir,
        "data_path": cfg.data_path,
        "window_size": cfg.window_size,
        "delta": cfg.delta,
        "batch_size": cfg.batch_size,
        "mask_budget": cfg.mask_budget,
        "n_active": cfg.mask_budget,
        "seed": cfg.seed,
    }
    sig = inspect.signature(external_load_data)
    kwargs = {name: value for name, value in kwargs_map.items() if name in sig.parameters}
    result = external_load_data(**kwargs)

    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise ValueError("load_data() must return at least train/val/test DataLoaders")

    train_loader, val_loader, test_loader = result[:3]
    meta = {"source": "external_load_data"}
    if len(result) >= 4 and isinstance(result[3], dict):
        meta.update(result[3])
    return train_loader, val_loader, test_loader, meta


def _load_from_numpy(data_dir: Path, delta: int) -> Optional[Dict[str, Any]]:
    """Load aligned arrays from .npy files when available."""
    full_data_path = data_dir / "full_data.npy"
    tte_path = data_dir / "tte_labels.npy"
    y_path = data_dir / "y_labels.npy"

    if not (full_data_path.exists() and tte_path.exists()):
        return None

    data = np.load(full_data_path).astype(np.float32)
    tte_labels = np.load(tte_path).astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"full_data.npy must be 2D, got shape={data.shape}")
    if tte_labels.ndim != 1 or len(tte_labels) != len(data):
        raise ValueError("tte_labels.npy must have shape (T,) matching full_data.npy")

    cls_labels = (tte_labels <= float(delta)).astype(np.float32)

    if y_path.exists():
        y_labels = np.load(y_path).astype(np.float32)
        if y_labels.shape == cls_labels.shape and not np.array_equal(y_labels, cls_labels):
            warnings.warn(
                "y_labels.npy does not match labels derived from tte_labels.npy and "
                f"delta={delta}; using the derived labels to keep the experiment consistent."
            )

    return {
        "data_norm": data,
        "cls_labels": cls_labels,
        "tte_labels": tte_labels,
        "source": "npy",
    }


def _load_from_excel(cfg: CompareConfig) -> Dict[str, Any]:
    """Fallback to the project's Excel preprocessing pipeline."""
    if load_and_clean is None or normalize is None or generate_labels is None:
        raise FileNotFoundError(
            "No compatible .npy files found, and dataset.py preprocessing helpers are unavailable."
        )

    candidate_paths: List[Path] = []
    if cfg.data_path:
        candidate_paths.append(Path(cfg.data_path))
    candidate_paths.append(Path(cfg.data_dir) / "data.xlsx")
    candidate_paths.append(ROOT_DIR / "data.xlsx")

    excel_path = next((path for path in candidate_paths if path.exists()), None)
    if excel_path is None:
        raise FileNotFoundError(
            "Could not locate full_data.npy/tte_labels.npy or a fallback data.xlsx file."
        )

    data_raw, cats, names = load_and_clean(
        str(excel_path),
        auto_reverse=getattr(cfg, "auto_reverse_experiments", True),
    )
    data_norm, _, _ = normalize(data_raw)
    _, tte_labels = generate_labels(
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
    cls_labels = (tte_labels <= float(cfg.delta)).astype(np.float32)
    return {
        "data_norm": data_norm.astype(np.float32),
        "cls_labels": cls_labels.astype(np.float32),
        "tte_labels": tte_labels.astype(np.float32),
        "source": "excel",
        "data_path": str(excel_path),
    }


def load_experiment_arrays(cfg: CompareConfig) -> Dict[str, Any]:
    """Load the underlying arrays once so each model can rebuild identical loaders."""
    npy_result = _load_from_numpy(Path(cfg.data_dir), cfg.delta)
    if npy_result is not None:
        return npy_result
    return _load_from_excel(cfg)


def build_dataloaders_from_arrays(
    arrays: Dict[str, Any],
    cfg: CompareConfig,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create train/val/test DataLoaders from loaded arrays."""
    seed = cfg.seed if seed is None else seed
    data = arrays["data_norm"]
    cls_labels = arrays["cls_labels"]
    tte_labels = arrays["tte_labels"]

    T, n_sensors = data.shape
    all_indices = np.arange(cfg.window_size - 1, T, dtype=np.int64)
    all_window_labels = cls_labels[all_indices]

    def _label_counts(indices: np.ndarray) -> Dict[float, int]:
        if len(indices) == 0:
            return {}
        uniq, cnt = np.unique(cls_labels[indices], return_counts=True)
        return {float(k): int(v) for k, v in zip(uniq, cnt)}

    def _build_time_split():
        train_end = int(T * cfg.train_ratio)
        val_end = int(T * (cfg.train_ratio + cfg.val_ratio))
        if train_end < cfg.window_size:
            raise ValueError("Train split is too short for the requested window size.")
        if val_end - max(0, train_end - cfg.window_size + 1) < cfg.window_size:
            raise ValueError("Time split is too short for the requested window size.")

        train_data = data[:train_end]
        train_cls = cls_labels[:train_end]
        train_tte = tte_labels[:train_end]

        val_start = max(0, train_end - cfg.window_size + 1)
        val_data = data[val_start:val_end]
        val_cls = cls_labels[val_start:val_end]
        val_tte = tte_labels[val_start:val_end]

        test_start = max(0, val_end - cfg.window_size + 1)
        test_data = data[test_start:]
        test_cls = cls_labels[test_start:]
        test_tte = tte_labels[test_start:]
        if len(test_data) < cfg.window_size:
            raise ValueError("Test split is too short for the requested window size.")

        train_dataset = WindowedRiskDataset(
            train_data, train_cls, train_tte,
            window_size=cfg.window_size,
            mask_budget=cfg.mask_budget,
            training=True,
            seed=seed,
        )
        val_dataset = WindowedRiskDataset(
            val_data, val_cls, val_tte,
            window_size=cfg.window_size,
            mask_budget=cfg.mask_budget,
            training=False,
            seed=seed + 10_000,
        )
        test_dataset = WindowedRiskDataset(
            test_data, test_cls, test_tte,
            window_size=cfg.window_size,
            mask_budget=cfg.mask_budget,
            training=False,
            seed=seed + 20_000,
        )
        meta = {
            "source": arrays.get("source", "unknown"),
            "n_sensors": n_sensors,
            "input_dim": 2 * n_sensors,
            "n_timesteps": T,
            "train_range": (0, train_end),
            "val_range": (train_end, val_end),
            "test_range": (val_end, T),
            "train_positive": int((train_cls[cfg.window_size - 1 :] > 0.5).sum()),
            "train_negative": int(len(train_cls[cfg.window_size - 1 :]) - (train_cls[cfg.window_size - 1 :] > 0.5).sum()),
            "min_pos_weight": float(cfg.min_pos_weight),
            "max_pos_weight": float(cfg.max_pos_weight),
            "mask_budget": cfg.mask_budget,
            "split_mode": "time",
            "train_label_counts": _label_counts(train_dataset.indices),
            "val_label_counts": _label_counts(val_dataset.indices + val_start),
            "test_label_counts": _label_counts(test_dataset.indices + test_start),
        }
        return train_dataset, val_dataset, test_dataset, meta

    def _build_stratified_split():
        stratify = all_window_labels if len(np.unique(all_window_labels)) > 1 else None
        train_idx, temp_idx = train_test_split(
            all_indices,
            train_size=cfg.train_ratio,
            shuffle=True,
            stratify=stratify,
            random_state=seed,
        )
        temp_ratio = cfg.val_ratio + cfg.test_ratio
        val_frac_of_temp = cfg.val_ratio / temp_ratio
        temp_labels = cls_labels[temp_idx]
        temp_stratify = temp_labels if len(np.unique(temp_labels)) > 1 else None
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_frac_of_temp,
            shuffle=True,
            stratify=temp_stratify,
            random_state=seed,
        )
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(test_idx)

        train_dataset = WindowedRiskDataset(
            data, cls_labels, tte_labels,
            window_size=cfg.window_size,
            mask_budget=cfg.mask_budget,
            training=True,
            seed=seed,
            indices=train_idx,
        )
        val_dataset = WindowedRiskDataset(
            data, cls_labels, tte_labels,
            window_size=cfg.window_size,
            mask_budget=cfg.mask_budget,
            training=False,
            seed=seed + 10_000,
            indices=val_idx,
        )
        test_dataset = WindowedRiskDataset(
            data, cls_labels, tte_labels,
            window_size=cfg.window_size,
            mask_budget=cfg.mask_budget,
            training=False,
            seed=seed + 20_000,
            indices=test_idx,
        )
        meta = {
            "source": arrays.get("source", "unknown"),
            "n_sensors": n_sensors,
            "input_dim": 2 * n_sensors,
            "n_timesteps": T,
            "train_range": None,
            "val_range": None,
            "test_range": None,
            "train_positive": int((cls_labels[train_idx] > 0.5).sum()),
            "train_negative": int(len(train_idx) - (cls_labels[train_idx] > 0.5).sum()),
            "min_pos_weight": float(cfg.min_pos_weight),
            "max_pos_weight": float(cfg.max_pos_weight),
            "mask_budget": cfg.mask_budget,
            "split_mode": "stratified",
            "train_label_counts": _label_counts(train_idx),
            "val_label_counts": _label_counts(val_idx),
            "test_label_counts": _label_counts(test_idx),
        }
        return train_dataset, val_dataset, test_dataset, meta

    if cfg.split_mode == "time":
        train_dataset, val_dataset, test_dataset, meta = _build_time_split()
    elif cfg.split_mode == "stratified":
        train_dataset, val_dataset, test_dataset, meta = _build_stratified_split()
    else:
        train_dataset, val_dataset, test_dataset, meta = _build_time_split()
        val_counts = meta["val_label_counts"]
        test_counts = meta["test_label_counts"]
        if len(val_counts) < 2 or len(test_counts) < 2:
            warnings.warn(
                "Time split produced a single-class validation/test split. "
                "Falling back to stratified window split for a meaningful architecture comparison."
            )
            train_dataset, val_dataset, test_dataset, meta = _build_stratified_split()

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, meta


def load_or_build_dataloaders(
    cfg: CompareConfig,
    arrays: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Public interface for this script.

    It first tries a compatible external load_data(); if unavailable, it falls
    back to local array loading and sliding-window dataset construction.
    """
    if cfg.data_protocol == "project":
        return build_project_dataloaders(cfg, seed=seed)

    if arrays is None:
        external = _try_external_load_data(cfg)
        if external is not None:
            return external
        arrays = load_experiment_arrays(cfg)
    return build_dataloaders_from_arrays(arrays=arrays, cfg=cfg, seed=seed)


def sanity_check_dataloaders(
    train_loader: DataLoader,
    meta: Dict[str, Any],
    cfg: CompareConfig,
    device: torch.device,
) -> None:
    """Cheap shape and mask-budget checks before the expensive training loop."""
    x, label, tte = next(iter(train_loader))
    n_sensors = int(meta["n_sensors"])
    assert x.ndim == 3, f"Expected 3D batch, got shape={tuple(x.shape)}"
    assert x.shape[1] == cfg.window_size, f"Expected window={cfg.window_size}, got {x.shape[1]}"
    assert x.shape[2] == 2 * n_sensors, f"Expected input_dim={2 * n_sensors}, got {x.shape[2]}"
    assert label.ndim == 1 and tte.ndim == 1

    mask_slice = x[:, 0, n_sensors:]
    active_counts = mask_slice.sum(dim=1)
    expected = min(cfg.mask_budget, n_sensors)
    if not torch.all(active_counts.eq(float(expected))):
        raise AssertionError(
            f"Each sample must activate exactly {expected} sensors, got {active_counts.tolist()}"
        )

    for model_name in cfg.model_names:
        model = build_model(model_name, n_sensors=n_sensors, cfg=cfg).to(device)
        with torch.no_grad():
            logit, tte_hat = model(x.to(device))
        assert logit.shape == label.to(device).shape, (
            f"{model_name} logit shape mismatch: {tuple(logit.shape)} vs {tuple(label.shape)}"
        )
        assert tte_hat.shape == tte.to(device).shape, (
            f"{model_name} tte_hat shape mismatch: {tuple(tte_hat.shape)} vs {tuple(tte.shape)}"
        )
        if torch.any(tte_hat < 0):
            raise AssertionError(f"{model_name} produced negative TTE values")


def compute_pos_weight(train_meta: Dict[str, Any]) -> float:
    """Match the project's BCE weighting strategy for class imbalance."""
    pos = train_meta.get("train_positive", 0)
    neg = train_meta.get("train_negative", 0)
    raw = neg / max(pos, 1)
    min_pw = float(train_meta.get("min_pos_weight", 0.1))
    max_pw = float(train_meta.get("max_pos_weight", 30.0))
    return float(min(max(raw, min_pw), max_pw))


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    """One training epoch shared by all backbones."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_tte = 0.0
    n_batches = 0

    for x, label, tte_true in dataloader:
        x = x.to(device)
        label = label.to(device)
        tte_true = tte_true.to(device)

        optimizer.zero_grad()
        logit, tte_hat = model(x)
        loss, cls_loss, tte_loss = criterion(logit, tte_hat, label, tte_true)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_tte += tte_loss.item()
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("inf"), "cls_loss": float("inf"), "tte_loss": float("inf")}

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "tte_loss": total_tte / n_batches,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    huber_delta: float = 1.0,
) -> Dict[str, float]:
    """Compute classification and TTE metrics on validation/test splits."""
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_tte_pred: List[torch.Tensor] = []
    all_tte_true: List[torch.Tensor] = []

    total_loss = 0.0
    total_cls = 0.0
    total_tte = 0.0
    n_batches = 0

    for x, label, tte_true in loader:
        x = x.to(device)
        label = label.to(device)
        tte_true = tte_true.to(device)

        logit, tte_hat = model(x)
        loss, cls_loss, tte_loss = criterion(logit, tte_hat, label, tte_true)

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_tte += tte_loss.item()
        n_batches += 1

        all_logits.append(logit.detach().cpu())
        all_labels.append(label.detach().cpu())
        all_tte_pred.append(tte_hat.detach().cpu())
        all_tte_true.append(tte_true.detach().cpu())

    if n_batches == 0:
        return {"loss": float("inf")}

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    tte_pred = torch.cat(all_tte_pred)
    tte_true = torch.cat(all_tte_true)

    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(np.int64)
    labels_np = labels.numpy().astype(np.int64)

    f1 = f1_score(labels_np, preds, zero_division=0)
    recall = recall_score(labels_np, preds, zero_division=0)
    precision = precision_score(labels_np, preds, zero_division=0)
    if len(np.unique(labels_np)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(labels_np, probs)
    if not np.isfinite(auc):
        auc = 0.0

    tte_huber = F.huber_loss(tte_pred, tte_true, reduction="mean", delta=huber_delta).item()
    tte_mse = F.mse_loss(tte_pred, tte_true, reduction="mean").item()
    tte_mae = F.l1_loss(tte_pred, tte_true, reduction="mean").item()

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "tte_loss": total_tte / n_batches,
        "auc": float(auc),
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "tte_huber": float(tte_huber),
        "tte_mse": float(tte_mse),
        "tte_mae": float(tte_mae),
    }


def _to_jsonable(obj: Any):
    """Recursively convert common runtime objects into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CompareConfig,
    device: torch.device,
    pos_weight: float,
) -> Tuple[Dict[str, float], str]:
    """Train a single backbone with early stopping and per-model checkpointing."""
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    seed_suffix = f"seed{int(cfg.seed)}"
    ckpt_path = save_dir / f"{model_name}_{seed_suffix}_best.pt"
    history_path = save_dir / f"{model_name}_{seed_suffix}_history.json"

    criterion = MultiTaskLoss(
        lambda_tte=cfg.lambda_tte,
        huber_delta=cfg.huber_delta,
        pos_weight=pos_weight,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    patience_counter = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_f1": [],
        "val_recall": [],
        "val_tte_huber": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            grad_clip=cfg.grad_clip,
        )
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=cfg.alarm_threshold,
            huber_delta=cfg.huber_delta,
        )

        history["train_loss"].append(float(train_metrics["loss"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_auc"].append(float(val_metrics["auc"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_recall"].append(float(val_metrics["recall"]))
        history["val_tte_huber"].append(float(val_metrics["tte_huber"]))

        if epoch % cfg.print_every == 0:
            elapsed = time.time() - epoch_start
            print(
                f"[{model_name.upper():11s}] epoch {epoch:02d}/{cfg.epochs} "
                f"{elapsed:5.1f}s | train={train_metrics['loss']:.4f} "
                f"val={val_metrics['loss']:.4f} auc={val_metrics['auc']:.4f} "
                f"f1={val_metrics['f1']:.4f} recall={val_metrics['recall']:.4f} "
                f"tte_huber={val_metrics['tte_huber']:.4f}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_metrics = {k: float(v) for k, v in val_metrics.items()}
            patience_counter = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": best_metrics,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"[{model_name.upper():11s}] early stop at epoch {epoch}")
                break

    save_json(history_path, history)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return best_metrics, str(ckpt_path)


def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    """Pretty-print the final comparison table with a tabulate fallback."""
    rows = []
    for model_name in sorted(results.keys()):
        metrics = results[model_name]
        rows.append(
            [
                model_name,
                f"{metrics['auc']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['tte_huber']:.4f}",
            ]
        )

    headers = ["Model", "AUC", "F1", "Recall", "Huber"]
    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="github")
    except ImportError:
        widths = [max(len(str(cell)) for cell in [header] + [row[i] for row in rows]) for i, header in enumerate(headers)]
        header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
        sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
        row_lines = [" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
        return "\n".join([header_line, sep_line] + row_lines)


def save_results_csv(path: Path, results: Dict[str, Dict[str, float]]) -> None:
    """Save the test metrics in a simple machine-readable format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "auc", "f1", "recall", "tte_huber", "tte_mse", "tte_mae"])
        for model_name in sorted(results.keys()):
            metrics = results[model_name]
            writer.writerow(
                [
                    model_name,
                    metrics["auc"],
                    metrics["f1"],
                    metrics["recall"],
                    metrics["tte_huber"],
                    metrics["tte_mse"],
                    metrics["tte_mae"],
                ]
            )


def summarize_repeated_results(
    per_seed_results: Dict[int, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate repeated runs as mean/std per model and metric."""
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for _seed, seed_results in per_seed_results.items():
        for model_name, metrics in seed_results.items():
            model_bucket = by_model.setdefault(model_name, {})
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(float(value)):
                    model_bucket.setdefault(key, []).append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for model_name, metrics in by_model.items():
        summary[model_name] = {}
        for key, values in metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            summary[model_name][f"{key}_mean"] = float(arr.mean()) if len(arr) else 0.0
            summary[model_name][f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            summary[model_name][f"{key}_n"] = int(len(arr))
    return summary


def format_summary_table(summary: Dict[str, Dict[str, float]]) -> str:
    rows = []
    for model_name in sorted(summary.keys()):
        metrics = summary[model_name]
        rows.append(
            [
                model_name,
                f"{metrics.get('auc_mean', 0.0):.4f} +- {metrics.get('auc_std', 0.0):.4f}",
                f"{metrics.get('f1_mean', 0.0):.4f} +- {metrics.get('f1_std', 0.0):.4f}",
                f"{metrics.get('recall_mean', 0.0):.4f} +- {metrics.get('recall_std', 0.0):.4f}",
                f"{metrics.get('tte_mae_mean', 0.0):.3f} +- {metrics.get('tte_mae_std', 0.0):.3f}",
            ]
        )

    headers = ["Model", "AUC", "F1", "Recall", "TTE MAE"]
    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="github")
    except ImportError:
        widths = [max(len(str(cell)) for cell in [header] + [row[i] for row in rows]) for i, header in enumerate(headers)]
        header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
        sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
        row_lines = [" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
        return "\n".join([header_line, sep_line] + row_lines)


def save_summary_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_roots = ["auc", "f1", "recall", "precision", "tte_huber", "tte_mse", "tte_mae", "loss"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["model"]
        for metric in metric_roots:
            header.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
        writer.writerow(header)
        for model_name in sorted(summary.keys()):
            row = [model_name]
            metrics = summary[model_name]
            for metric in metric_roots:
                row.extend([
                    metrics.get(f"{metric}_mean", 0.0),
                    metrics.get(f"{metric}_std", 0.0),
                    metrics.get(f"{metric}_n", 0),
                ])
            writer.writerow(row)


def run_comparison_once(cfg: CompareConfig, arrays: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """End-to-end experiment loop across backbones for one seed."""
    device = select_device(cfg.device)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    train_loader, _, _, meta = load_or_build_dataloaders(
        cfg=cfg, arrays=arrays, seed=cfg.seed
    )
    sanity_check_dataloaders(train_loader, meta, cfg, device)

    print("=" * 72)
    print("Predictor architecture comparison")
    print("=" * 72)
    print(
        f"source={meta['source']} | timesteps={meta['n_timesteps']} | sensors={meta['n_sensors']} | "
        f"window={cfg.window_size} | mask_budget={cfg.mask_budget} | delta={cfg.delta}"
    )
    print(
        f"split_mode={meta['split_mode']} | train={meta['train_range']} val={meta['val_range']} test={meta['test_range']} | "
        f"train_pos={meta['train_positive']} train_neg={meta['train_negative']}"
    )
    print(
        f"label_counts train={meta['train_label_counts']} "
        f"val={meta['val_label_counts']} test={meta['test_label_counts']}"
    )
    print("=" * 72)

    pos_weight = compute_pos_weight(meta)
    test_results: Dict[str, Dict[str, float]] = {}
    artifact_summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "meta": meta,
        "models": {},
    }

    for model_name in cfg.model_names:
        print(f"\n--- Training {model_name} ---")
        set_seed(cfg.seed)
        train_loader, val_loader, test_loader, meta = load_or_build_dataloaders(
            cfg=cfg, arrays=arrays, seed=cfg.seed
        )
        model = build_model(model_name, n_sensors=meta["n_sensors"], cfg=cfg).to(device)

        best_val_metrics, ckpt_path = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            pos_weight=pos_weight,
        )

        criterion = MultiTaskLoss(
            lambda_tte=cfg.lambda_tte,
            huber_delta=cfg.huber_delta,
            pos_weight=pos_weight,
        ).to(device)
        test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=cfg.alarm_threshold,
            huber_delta=cfg.huber_delta,
        )

        test_results[model_name] = {k: float(v) for k, v in test_metrics.items()}
        artifact_summary["models"][model_name] = {
            "checkpoint": ckpt_path,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_results[model_name],
        }
        print(
            f"[{model_name.upper():11s}] test auc={test_metrics['auc']:.4f} "
            f"f1={test_metrics['f1']:.4f} recall={test_metrics['recall']:.4f} "
            f"tte_huber={test_metrics['tte_huber']:.4f}"
        )

    print("\nFinal comparison")
    print(format_results_table(test_results))
    return test_results, artifact_summary


def run_comparison(cfg: CompareConfig) -> Dict[str, Dict[str, float]]:
    """Run one or more seeds and save single-seed plus aggregate artifacts."""
    arrays = load_experiment_arrays(cfg) if cfg.data_protocol == "legacy" else None
    per_seed_results: Dict[int, Dict[str, Dict[str, float]]] = {}
    per_seed_artifacts: Dict[str, Any] = {}

    for seed in cfg.seeds:
        print("\n" + "#" * 72)
        print(f"Predictor architecture comparison seed={seed}")
        print("#" * 72)
        seed_cfg = replace(cfg, seed=int(seed))
        seed_results, seed_artifact = run_comparison_once(seed_cfg, arrays=arrays)
        per_seed_results[int(seed)] = seed_results
        per_seed_artifacts[str(seed)] = seed_artifact

    summary = summarize_repeated_results(per_seed_results)
    payload = {
        "config": asdict(cfg),
        "per_seed": per_seed_artifacts,
        "summary": summary,
    }
    save_json(Path(cfg.save_dir) / "compare_predictors_results.json", payload)
    save_json(Path(cfg.save_dir) / "compare_predictors_summary.json", summary)
    if len(cfg.seeds) == 1:
        only_seed = int(cfg.seeds[0])
        save_results_csv(Path(cfg.save_dir) / "compare_predictors_results.csv", per_seed_results[only_seed])
    save_summary_csv(Path(cfg.save_dir) / "compare_predictors_summary.csv", summary)

    print("\nRepeated-run summary")
    print(format_summary_table(summary))
    return summary


def main() -> Dict[str, Dict[str, float]]:
    cfg = parse_args()
    return run_comparison(cfg)


if __name__ == "__main__":
    main()
