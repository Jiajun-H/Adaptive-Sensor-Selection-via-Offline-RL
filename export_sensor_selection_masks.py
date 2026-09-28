"""
Export policy-selected sensor masks and sensor-level summaries from checkpoints.

Run this script in the project environment with torch/numpy/pandas/matplotlib
installed, for example:

    python export_sensor_selection_masks.py --checkpoint-dir checkpoints/repeated/kfold_seed42_fold0

It loads the saved predictor and IQL policy, evaluates the held-out split, then
exports raw masks plus sensor-frequency, stage-frequency, and modality summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1")


def _configure_windows_dll_search_paths() -> None:
    """Make conda-installed PyTorch DLLs discoverable before importing torch."""
    if os.name != "nt":
        return
    prefix = Path(sys.exec_prefix)
    dll_dirs = [
        prefix / "Library" / "bin",
        prefix / "Lib" / "site-packages" / "torch" / "lib",
        prefix / "bin",
    ]
    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            continue
        try:
            os.add_dll_directory(str(dll_dir))
        except (AttributeError, FileNotFoundError, OSError):
            pass

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    missing_dirs = [str(dll_dir) for dll_dir in dll_dirs if dll_dir.exists() and str(dll_dir) not in path_parts]
    if missing_dirs:
        os.environ["PATH"] = os.pathsep.join(missing_dirs + path_parts)


_configure_windows_dll_search_paths()

# Import torch before numpy to match the main experiment scripts and avoid
# Windows DLL/OpenMP resolution conflicts in some conda environments.
try:
    import torch
except OSError as exc:
    raise SystemExit(
        "Failed to import PyTorch after configuring Windows DLL search paths.\n"
        "The script now adds the conda Library/bin, torch/lib, and bin paths "
        "before importing torch. If this still fails, the active PyTorch "
        "installation is likely incomplete or ABI-incompatible.\n\n"
        f"Original import error:\n{exc}"
    ) from exc

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from dataset import prepare_data_time_split
from evaluate import evaluate_with_policy_multi_experiments
from iql_networks import PolicyNetwork
from model import MaskAwareRiskPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export IQL sensor-selection masks from a checkpoint directory.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/repeated/kfold_seed42_fold0"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-eval-batch-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_from_results(checkpoint_dir: Path, device: str, policy_eval_batch_size: int) -> Config:
    cfg = Config()
    results_path = checkpoint_dir / "experiment_results.json"
    if results_path.exists():
        saved = load_json(results_path).get("config", {})
        cfg.n_active = int(saved.get("n_active", cfg.n_active))
        cfg.window_size = int(saved.get("window_size", cfg.window_size))
        cfg.delta = int(saved.get("delta", cfg.delta))
        cfg.alarm_threshold = float(saved.get("alarm_threshold", cfg.alarm_threshold))
        cfg.tune_alarm_threshold = False
        cfg.threshold_metric = str(saved.get("threshold_metric", cfg.threshold_metric))
        cfg.experiment_split_mode = str(saved.get("experiment_split_mode", cfg.experiment_split_mode))
        cfg.split_seed = int(saved.get("split_seed", cfg.split_seed))
        cfg.experiment_kfold_n_splits = int(saved.get("experiment_kfold_n_splits", cfg.experiment_kfold_n_splits))
        cfg.experiment_kfold_index = int(saved.get("experiment_kfold_index", cfg.experiment_kfold_index))
        cfg.policy_belief_mode = str(saved.get("policy_belief_mode", cfg.policy_belief_mode))
    cfg.save_dir = str(checkpoint_dir)
    cfg.fig_dir = str(checkpoint_dir)
    cfg.device = device
    cfg.offline_device = device
    cfg.data_num_workers = 0
    cfg.rl_num_workers = 0
    cfg.policy_eval_batch_size = max(1, int(policy_eval_batch_size))
    return cfg


def make_iql_policy_fn(pi_net: PolicyNetwork, predictor: MaskAwareRiskPredictor | None = None, belief_mode: str = "full"):
    state = {"mask": None}

    def policy_fn(predictor, obs_window, n_sensors, m, dev):
        if belief_mode == "previous_mask" and state["mask"] is not None:
            belief_mask = state["mask"]
        elif belief_mode == "previous_mask":
            belief_mask = np.zeros(n_sensors, dtype=np.float32)
            belief_mask[: min(m, n_sensors)] = 1.0
        else:
            belief_mask = np.ones(n_sensors, dtype=np.float32)
        obs_masked = obs_window * belief_mask[np.newaxis, :]
        mask_exp = np.tile(belief_mask, (obs_window.shape[0], 1))
        x = np.concatenate([obs_masked, mask_exp], axis=-1)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            h = predictor.encode_belief(x_t)
            mask_out = pi_net.select_top_m(h, m=m)
        selected = mask_out.detach().cpu().numpy().squeeze(0).astype(np.float32)
        state["mask"] = selected.copy()
        return selected

    policy_fn.reset = lambda: state.update({"mask": None})  # type: ignore[attr-defined]

    if belief_mode == "full" and predictor is not None:
        def batch_select(obs_windows, n_sensors, m, dev):
            obs_batch = np.asarray(obs_windows, dtype=np.float32)
            if obs_batch.ndim != 3:
                raise ValueError(f"Expected obs_windows with shape (B,W,N), got {obs_batch.shape}")
            batch_size, window_size, _ = obs_batch.shape
            belief_mask = np.ones((batch_size, n_sensors), dtype=np.float32)
            mask_exp = np.broadcast_to(
                belief_mask[:, np.newaxis, :],
                (batch_size, window_size, n_sensors),
            )
            x = np.concatenate([obs_batch, mask_exp], axis=-1)
            x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
            with torch.no_grad():
                h = predictor.encode_belief(x_t)
                mask_out = pi_net.select_top_m(h, m=m)
            return mask_out.detach().cpu().numpy().astype(np.float32)

        policy_fn.batch_select = batch_select  # type: ignore[attr-defined]

    return policy_fn


def load_models(cfg: Config, checkpoint_dir: Path, n_sensors: int, device: torch.device):
    predictor_ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    predictor = MaskAwareRiskPredictor(
        n_sensors=n_sensors,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lstm_dropout=cfg.lstm_dropout,
        head_dropout=cfg.head_dropout,
    ).to(device)
    predictor.load_state_dict(predictor_ckpt["model_state_dict"])
    predictor.eval()

    policy_ckpt = torch.load(checkpoint_dir / "policy_best.pt", map_location=device, weights_only=False)
    pi = PolicyNetwork(
        policy_ckpt.get("hidden_dim", cfg.hidden_dim),
        policy_ckpt.get("n_sensors", n_sensors),
        cfg.iql_hidden_dim,
    ).to(device)
    pi.load_state_dict(policy_ckpt["pi_net"])
    pi.eval()
    return predictor, pi


def rows_to_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aligned_stage_arrays(test_experiments: list[dict], window: int) -> tuple[np.ndarray, np.ndarray]:
    ttes = []
    labels = []
    for exp in test_experiments:
        if len(exp["ttes"]) >= window:
            ttes.append(np.asarray(exp["ttes"][window - 1 :], dtype=np.float32))
            labels.append(np.asarray(exp["cls_labels"][window - 1 :], dtype=np.float32))
    if not ttes:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(ttes), np.concatenate(labels)


def stage_name(tte: float, label: float, delta: int) -> str:
    if tte <= 0.0 and label > 0.5:
        return "danger"
    if 0.0 < tte <= float(delta):
        return "warning_horizon"
    return "pre_warning"


def export_summaries(
    output_dir: Path,
    masks: np.ndarray,
    sensor_names: list[str],
    sensor_types: list[str],
    ttes: np.ndarray,
    labels: np.ndarray,
    delta: int,
) -> dict:
    n_steps, n_sensors = masks.shape
    freq = masks.mean(axis=0) if n_steps > 0 else np.zeros(n_sensors, dtype=np.float32)
    order = np.argsort(freq)[::-1]

    rows = []
    for rank, idx in enumerate(order, 1):
        rows.append(
            {
                "rank": rank,
                "sensor_index": int(idx),
                "sensor_name": sensor_names[idx] if idx < len(sensor_names) else f"S{idx}",
                "sensor_type": sensor_types[idx] if idx < len(sensor_types) else "unknown",
                "selection_frequency": f"{float(freq[idx]):.6f}",
                "selected_steps": int(masks[:, idx].sum()),
            }
        )
    rows_to_csv(
        output_dir / "iql_sensor_selection_frequency.csv",
        ["rank", "sensor_index", "sensor_name", "sensor_type", "selection_frequency", "selected_steps"],
        rows,
    )

    stages = np.array([stage_name(float(t), float(y), delta) for t, y in zip(ttes, labels)], dtype=object)
    stage_rows = []
    for stage in ["pre_warning", "warning_horizon", "danger"]:
        mask_stage = stages == stage
        denom = int(mask_stage.sum())
        if denom == 0:
            continue
        stage_freq = masks[mask_stage].mean(axis=0)
        for idx in order:
            stage_rows.append(
                {
                    "stage": stage,
                    "sensor_index": int(idx),
                    "sensor_name": sensor_names[idx] if idx < len(sensor_names) else f"S{idx}",
                    "sensor_type": sensor_types[idx] if idx < len(sensor_types) else "unknown",
                    "selection_frequency": f"{float(stage_freq[idx]):.6f}",
                    "stage_steps": denom,
                }
            )
    rows_to_csv(
        output_dir / "iql_sensor_selection_by_stage.csv",
        ["stage", "sensor_index", "sensor_name", "sensor_type", "selection_frequency", "stage_steps"],
        stage_rows,
    )

    modality_rows = []
    for stage in ["overall", "pre_warning", "warning_horizon", "danger"]:
        if stage == "overall":
            sub = masks
        else:
            sub = masks[stages == stage]
        if len(sub) == 0:
            continue
        selected_total = float(sub.sum())
        for sensor_type in sorted(set(sensor_types)):
            idx = [i for i, t in enumerate(sensor_types) if t == sensor_type]
            selected = float(sub[:, idx].sum()) if idx else 0.0
            modality_rows.append(
                {
                    "stage": stage,
                    "sensor_type": sensor_type,
                    "selected_slots": f"{selected:.0f}",
                    "slot_share": f"{selected / selected_total:.6f}" if selected_total > 0 else "0.000000",
                    "available_sensors": len(idx),
                }
            )
    rows_to_csv(
        output_dir / "iql_modality_selection_summary.csv",
        ["stage", "sensor_type", "selected_slots", "slot_share", "available_sensors"],
        modality_rows,
    )

    return {"top_indices": order.tolist(), "frequency": freq, "stages": stages}


def plot_figures(output_dir: Path, masks: np.ndarray, summary: dict, sensor_names: list[str], sensor_types: list[str], top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_indices = summary["top_indices"][: min(top_k, masks.shape[1])]
    sub = masks[:, top_indices].T
    labels = [
        f"{sensor_names[i] if i < len(sensor_names) else f'S{i}'} ({sensor_types[i] if i < len(sensor_types) else 'unknown'})"
        for i in top_indices
    ]

    fig_h = max(5.5, 0.24 * len(top_indices))
    fig, ax = plt.subplots(figsize=(13.0, fig_h))
    im = ax.imshow(sub, aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title("IQL Sensor Selection Heatmap (Top-ranked Sensors)")
    ax.set_xlabel("Evaluation time step")
    ax.set_ylabel("Sensor")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Selected")
    fig.tight_layout()
    fig.savefig(output_dir / "iql_sensor_selection_heatmap_topk.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / "iql_sensor_selection_heatmap_topk.pdf", bbox_inches="tight")
    plt.close(fig)

    freq = summary["frequency"]
    fig, ax = plt.subplots(figsize=(8.2, max(5.0, 0.18 * len(top_indices))))
    vals = [freq[i] for i in top_indices][::-1]
    names = [labels[j] for j in range(len(labels))][::-1]
    colors = ["#2F65D9"] * len(vals)
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Selection frequency")
    ax.set_title("Most Frequently Selected Sensors")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "iql_sensor_frequency_topk.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / "iql_sensor_frequency_topk.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir or Path("figures/sensor_selection_paper") / checkpoint_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = configure_from_results(checkpoint_dir, str(device), args.policy_eval_batch_size)
    print(f"[export] checkpoint_dir={checkpoint_dir}")
    print(f"[export] output_dir={output_dir}")
    print(f"[export] device={device}, policy_eval_batch_size={cfg.policy_eval_batch_size}")
    print("[export] loading and aligning experiment data...")
    t_data = time.perf_counter()
    data_info = prepare_data_time_split(cfg)
    print(f"[export] data ready in {time.perf_counter() - t_data:.1f}s")
    n_sensors = int(data_info["n_sensors"])
    print(f"[export] aligned sensors={n_sensors}, test_experiments={len(data_info['test_experiments'])}")
    print("[export] loading predictor and IQL policy checkpoints...")
    t_model = time.perf_counter()
    predictor, pi = load_models(cfg, checkpoint_dir, n_sensors, device)
    print(f"[export] checkpoints loaded in {time.perf_counter() - t_model:.1f}s")
    policy_fn = make_iql_policy_fn(pi, predictor=predictor, belief_mode=str(getattr(cfg, "policy_belief_mode", "full")))

    print("[export] evaluating policy and collecting masks...")
    t_eval = time.perf_counter()
    metrics = evaluate_with_policy_multi_experiments(
        predictor,
        policy_fn,
        data_info["test_experiments"],
        cfg,
        str(device),
    )
    print(f"[export] masks collected in {time.perf_counter() - t_eval:.1f}s")
    masks = np.asarray(metrics["masks"], dtype=np.float32)
    sensor_names = list(data_info.get("sensor_names", [f"S{i}" for i in range(n_sensors)]))
    sensor_types = list(data_info.get("sensor_types", ["unknown"] * n_sensors))
    ttes, labels = aligned_stage_arrays(data_info["test_experiments"], cfg.window_size)

    np.savez_compressed(
        output_dir / "iql_sensor_selection_masks.npz",
        masks=masks,
        ttes=ttes,
        labels=labels,
        sensor_names=np.asarray(sensor_names, dtype=object),
        sensor_types=np.asarray(sensor_types, dtype=object),
    )
    print("[export] writing mask arrays and CSV summaries...")
    t_summary = time.perf_counter()
    summary = export_summaries(output_dir, masks, sensor_names, sensor_types, ttes, labels, cfg.delta)
    print(f"[export] summaries written in {time.perf_counter() - t_summary:.1f}s")
    if not args.no_figures:
        print("[export] rendering figures...")
        t_fig = time.perf_counter()
        plot_figures(output_dir, masks, summary, sensor_names, sensor_types, args.top_k)
        print(f"[export] figures rendered in {time.perf_counter() - t_fig:.1f}s")

    metadata = {
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "n_sensors": n_sensors,
        "n_steps": int(masks.shape[0]),
        "n_active": int(cfg.n_active),
        "top_10_sensors": [
            {
                "rank": i + 1,
                "sensor_index": int(idx),
                "sensor_name": sensor_names[idx] if idx < len(sensor_names) else f"S{idx}",
                "sensor_type": sensor_types[idx] if idx < len(sensor_types) else "unknown",
                "selection_frequency": float(summary["frequency"][idx]),
            }
            for i, idx in enumerate(summary["top_indices"][:10])
        ],
    }
    (output_dir / "iql_sensor_selection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[export] saved masks and summaries to {output_dir}")
    print(f"[export] total elapsed {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
