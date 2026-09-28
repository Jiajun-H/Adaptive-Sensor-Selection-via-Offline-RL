"""Reproducible efficiency benchmark for predictor + offline-RL policies.

The online path is measured end to end on CPU:
raw (W, N) history + current mask -> masked model input -> belief encoder ->
policy logits -> Top-M sensor indices. Q/V networks are counted only as policy
training cost because they are not loaded for deployment.

FLOPs use the common MAC=2-FLOPs convention and count dense, recurrent,
attention, and convolutional arithmetic. Normalization, activations, memory
movement, and Top-K comparisons are excluded and the convention is written to
the JSON output so reported values remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import numpy as np

from compare_predictors import CompareConfig, TemporalBlock, build_model
from config import Config
from iql_networks import PolicyNetwork, QNetwork, VNetwork


POLICY_FILES = {
    "iql": "policy_best.pt",
    "cql": "policy_cql.pt",
    "bc": "policy_bc.pt",
}


class DecisionPipeline(nn.Module):
    """Deployment path from a raw history window to Top-M sensor indices."""

    def __init__(self, predictor: nn.Module, policy: PolicyNetwork, n_active: int):
        super().__init__()
        self.predictor = predictor
        self.policy = policy
        self.n_active = int(n_active)

    def forward(self, observations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask_window = mask.unsqueeze(1).expand(-1, observations.shape[1], -1)
        model_input = torch.cat([observations * mask_window, mask_window], dim=-1)
        belief = self.predictor.encode_belief(model_input)
        logits = self.policy(belief)
        return logits.topk(k=self.n_active, dim=-1).indices


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def module_tensor_bytes(modules: Iterable[nn.Module]) -> int:
    """Exact parameter and buffer bytes resident in the supplied modules."""
    seen = set()
    total = 0
    for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
            key = (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            if key in seen:
                continue
            seen.add(key)
            total += tensor.untyped_storage().nbytes()
    return int(total)


def _linear_flops(module: nn.Linear, tokens: int = 1) -> int:
    return int(2 * tokens * module.in_features * module.out_features)


def policy_flops(policy: PolicyNetwork) -> int:
    return sum(_linear_flops(module) for module in policy.modules() if isinstance(module, nn.Linear))


def _conv1d_flops(conv: nn.Conv1d, input_length: int) -> Tuple[int, int]:
    kernel = int(conv.kernel_size[0])
    stride = int(conv.stride[0])
    padding = int(conv.padding[0])
    dilation = int(conv.dilation[0])
    output_length = (
        input_length + 2 * padding - dilation * (kernel - 1) - 1
    ) // stride + 1
    operations = (
        2
        * output_length
        * conv.out_channels
        * (conv.in_channels // conv.groups)
        * kernel
    )
    return int(operations), int(output_length)


def _pool1d_output_length(pool: nn.MaxPool1d, input_length: int) -> int:
    kernel = int(pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0])
    stride_value = pool.stride if pool.stride is not None else kernel
    stride = int(stride_value if isinstance(stride_value, int) else stride_value[0])
    padding = int(pool.padding if isinstance(pool.padding, int) else pool.padding[0])
    dilation = int(pool.dilation if isinstance(pool.dilation, int) else pool.dilation[0])
    return int(
        (input_length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    )


def _transformer_stack_flops(layers: Iterable[nn.Module], tokens: int, d_model: int) -> int:
    total = 0
    for layer in layers:
        ff_dim = int(layer.linear1.out_features)
        # Q/K/V projections, output projection, QK^T, attention-value, and FFN.
        total += 8 * tokens * d_model * d_model
        total += 4 * tokens * tokens * d_model
        total += 4 * tokens * d_model * ff_dim
    return int(total)


def predictor_encoder_flops(name: str, predictor: nn.Module, window_size: int) -> int:
    """Count the arithmetic used by encode_belief for batch size one."""
    name = name.lower()
    if name == "lstm":
        backbone = predictor.backbone
        input_linear = backbone.input_proj[0]
        total = _linear_flops(input_linear, tokens=window_size)
        lstm = backbone.lstm
        input_size = int(lstm.input_size)
        hidden = int(lstm.hidden_size)
        for layer in range(int(lstm.num_layers)):
            layer_input = input_size if layer == 0 else hidden
            total += window_size * 8 * hidden * (layer_input + hidden)
        return int(total)

    if name == "cnn":
        total = 0
        length = int(window_size)
        for module in predictor.features:
            if isinstance(module, nn.Conv1d):
                flops, length = _conv1d_flops(module, length)
                total += flops
            elif isinstance(module, nn.MaxPool1d):
                length = _pool1d_output_length(module, length)
        return int(total)

    if name == "transformer":
        tokens = int(window_size)
        d_model = int(predictor.feature_dim)
        total = _linear_flops(predictor.input_proj, tokens=tokens)
        total += _transformer_stack_flops(predictor.encoder.layers, tokens, d_model)
        return int(total)

    if name == "tcn":
        total = 0
        length = int(window_size)
        for block in predictor.network:
            if not isinstance(block, TemporalBlock):
                continue
            conv1 = block.net[0]
            conv2 = block.net[4]
            flops, _ = _conv1d_flops(conv1, length)
            total += flops
            # Chomp restores the original sequence length before the second convolution.
            flops, _ = _conv1d_flops(conv2, length)
            total += flops
            if isinstance(block.downsample, nn.Conv1d):
                flops, _ = _conv1d_flops(block.downsample, length)
                total += flops
        return int(total)

    if name in {"patchtst", "patch_tst", "patch"}:
        tokens = int(predictor.n_patches)
        d_model = int(predictor.feature_dim)
        total = _linear_flops(predictor.patch_proj, tokens=tokens)
        total += _transformer_stack_flops(predictor.encoder.layers, tokens, d_model)
        return int(total)

    raise ValueError(f"FLOP counting is not implemented for backbone {name!r}")


def decision_flops(
    name: str,
    predictor: nn.Module,
    policy: PolicyNetwork,
    window_size: int,
    n_sensors: int,
) -> int:
    masking_flops = int(window_size * n_sensors)
    return masking_flops + predictor_encoder_flops(name, predictor, window_size) + policy_flops(policy)


def benchmark_latency(
    pipeline: DecisionPipeline,
    window_size: int,
    n_sensors: int,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    pipeline = pipeline.cpu().eval()
    observations = torch.randn(1, window_size, n_sensors, dtype=torch.float32)
    mask = torch.zeros(1, n_sensors, dtype=torch.float32)
    mask[:, : pipeline.n_active] = 1.0

    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            pipeline(observations, mask)
        samples = np.empty(max(1, repeats), dtype=np.float64)
        for index in range(len(samples)):
            start = time.perf_counter_ns()
            pipeline(observations, mask)
            samples[index] = (time.perf_counter_ns() - start) / 1e6

    return {
        "latency_mean_ms": float(samples.mean()),
        "latency_median_ms": float(np.median(samples)),
        "latency_std_ms": float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_min_ms": float(samples.min()),
        "latency_max_ms": float(samples.max()),
    }


def _state_dict_from_checkpoint(checkpoint: dict) -> Optional[dict]:
    for key in ("model_state_dict", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return None


def load_predictor_weights(
    predictor: nn.Module,
    name: str,
    checkpoint_dir: Path,
    trained_model_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    if trained_model_dir is not None:
        candidates = sorted((trained_model_dir / "predictor").glob(f"{name}*_best.pt"))
    elif name == "lstm":
        candidates = [checkpoint_dir / "best_model.pt"]
    else:
        compare_dir = checkpoint_dir / "compare_predictors"
        candidates = sorted(compare_dir.glob(f"{name}*_best.pt"))
    for path in candidates:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = _state_dict_from_checkpoint(checkpoint)
        if state is None:
            continue
        target = (
            predictor.backbone
            if trained_model_dir is None and name == "lstm" and hasattr(predictor, "backbone")
            else predictor
        )
        try:
            target.load_state_dict(state, strict=True)
        except RuntimeError:
            continue
        return True, str(path)
    return False, ""


def build_policy(
    policy_name: str,
    feature_dim: int,
    n_sensors: int,
    default_mlp_dim: int,
    checkpoint_dir: Path,
    load_checkpoint: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[PolicyNetwork, bool, str]:
    path = checkpoint_path or (checkpoint_dir / POLICY_FILES[policy_name])
    state = None
    mlp_dim = int(default_mlp_dim)
    if load_checkpoint and path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("pi_net")
        if isinstance(state, dict) and "net.0.weight" in state:
            mlp_dim = int(state["net.0.weight"].shape[0])
    policy = PolicyNetwork(feature_dim, n_sensors, mlp_dim)
    if isinstance(state, dict):
        try:
            policy.load_state_dict(state, strict=True)
            return policy, True, str(path)
        except RuntimeError:
            pass
    return policy, False, ""


def resolve_n_sensors(requested: Optional[int], cfg: Config, checkpoint_dir: Path) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    if cfg.n_sensors > 0:
        return int(cfg.n_sensors)
    path = checkpoint_dir / "best_model.pt"
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("n_sensors", 0)) > 0:
            return int(checkpoint["n_sensors"])
        state = _state_dict_from_checkpoint(checkpoint) or {}
        weight = state.get("input_proj.0.weight")
        if weight is not None:
            return int(weight.shape[1] // 2)
    raise ValueError("Cannot infer n_sensors; pass --n-sensors explicitly.")


def policy_training_cost(
    policy_name: str,
    feature_dim: int,
    n_sensors: int,
    mlp_dim: int,
) -> Dict[str, int]:
    policy = PolicyNetwork(feature_dim, n_sensors, mlp_dim)
    policy_params = count_parameters(policy)
    if policy_name == "iql":
        q_net = QNetwork(feature_dim, n_sensors, mlp_dim)
        v_net = VNetwork(feature_dim, mlp_dim)
        trainable = policy_params + count_parameters(q_net) + count_parameters(v_net)
        resident = trainable
    elif policy_name == "cql":
        q_net = QNetwork(feature_dim, n_sensors, mlp_dim)
        trainable = policy_params + count_parameters(q_net)
        resident = policy_params + 2 * count_parameters(q_net)
    else:
        trainable = resident = policy_params
    return {
        "policy_trainable_params": int(trainable),
        "policy_training_resident_params": int(resident),
    }


def read_history(checkpoint_dir: Path, policy_name: str) -> dict:
    path = checkpoint_dir / f"{policy_name}_history.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_history_path(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate_convergence(history: dict) -> Dict[str, Optional[float]]:
    values = np.asarray(history.get("val_action_f1", []), dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).any():
        return {
            "convergence_epoch": None,
            "convergence_updates": None,
            "convergence_val_f1": None,
        }
    epochs = np.asarray(history.get("epoch", np.arange(1, len(values) + 1)), dtype=np.float64)
    updates = np.asarray(history.get("train_updates", epochs), dtype=np.float64)
    index = int(np.nanargmax(values))
    return {
        "convergence_epoch": float(epochs[index]),
        "convergence_updates": float(updates[index]),
        "convergence_val_f1": float(values[index]),
    }


def plot_learning_curves(histories: Dict[str, dict], output_dir: Path) -> List[str]:
    available = {
        name: history
        for name, history in histories.items()
        if len(history.get("val_action_f1", [])) > 0
    }
    if not available:
        print("[Plot] no validation curves found; retrain policies to populate them.")
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    colors = {"iql": "#0072B2", "cql": "#D55E00", "bc": "#6B6B6B"}
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.75))
    for name, history in available.items():
        steps = history.get("train_updates", history.get("epoch", []))
        axes[0].plot(
            steps,
            history["val_action_f1"],
            label=name.upper(),
            color=colors[name],
            lw=1.5,
            marker="o",
            ms=3,
        )
    axes[0].set_xlabel("Training updates")
    axes[0].set_ylabel("Validation action F1")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    axes[0].text(-0.16, 1.03, "a", transform=axes[0].transAxes, fontsize=8, fontweight="bold")

    q_plotted = False
    for name in ("iql", "cql"):
        history = available.get(name, {})
        values = history.get("val_estimated_q", [])
        if values:
            steps = history.get("train_updates", history.get("epoch", []))
            axes[1].plot(
                steps,
                values,
                label=name.upper(),
                color=colors[name],
                lw=1.5,
                marker="o",
                ms=3,
            )
            q_plotted = True
    axes[1].set_xlabel("Training updates")
    axes[1].set_ylabel("Validation estimated Q")
    if q_plotted:
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No critic validation history", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].text(-0.16, 1.03, "b", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()

    stem = output_dir / "training_efficiency_curves"
    outputs = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    print(f"[Plot] saved learning curves to {stem}.[svg|pdf|tiff|png]")
    return outputs


def cpu_description() -> str:
    description = platform.processor().strip()
    if description:
        return description
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def write_csv(rows: Sequence[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model efficiency on CPU")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--backbone-run-dir",
        type=Path,
        default=None,
        help="Run directory containing one fully trained subdirectory per IQL backbone.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("efficiency_results"))
    parser.add_argument("--n-sensors", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--n-active", type=int, default=None)
    parser.add_argument(
        "--backbones",
        type=str,
        default="lstm,cnn,tcn,transformer,patchtst",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--decision-interval-ms", type=float, default=5000.0)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    checkpoint_dir = args.checkpoint_dir.resolve()
    backbone_run_dir = args.backbone_run_dir.resolve() if args.backbone_run_dir else None
    n_sensors = resolve_n_sensors(args.n_sensors, cfg, checkpoint_dir)
    window_size = int(args.window_size or cfg.window_size)
    n_active = min(n_sensors, int(args.n_active or cfg.n_active or 20))
    backbones = [name.strip().lower() for name in args.backbones.split(",") if name.strip()]
    supported_backbones = {"lstm", "cnn", "tcn", "transformer", "patchtst"}
    unsupported = sorted(set(backbones) - supported_backbones)
    if unsupported:
        raise ValueError(f"Unsupported efficiency backbones: {unsupported}")

    torch.set_num_threads(max(1, int(args.num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    compare_cfg = CompareConfig()
    compare_cfg.window_size = window_size
    compare_cfg.hidden_dim = cfg.hidden_dim
    compare_cfg.num_layers = cfg.num_layers
    compare_cfg.lstm_dropout = cfg.lstm_dropout
    compare_cfg.head_dropout = cfg.head_dropout

    methods = [(name, "iql") for name in backbones]
    if "lstm" in backbones:
        methods.extend([("lstm", "cql"), ("lstm", "bc")])

    histories = {name: read_history(checkpoint_dir, name) for name in POLICY_FILES}
    rows: List[dict] = []
    for backbone_name, policy_name in methods:
        trained_model_dir = (
            backbone_run_dir / backbone_name
            if policy_name == "iql" and backbone_run_dir is not None
            else None
        )
        required_backbone_artifacts = (
            trained_model_dir / "policy_best.pt",
            trained_model_dir / "iql_history.json",
        ) if trained_model_dir is not None else ()
        if trained_model_dir is not None and (
            not trained_model_dir.exists()
            or not all(path.exists() for path in required_backbone_artifacts)
        ):
            trained_model_dir = None
        predictor = build_model(backbone_name, n_sensors, compare_cfg).cpu().eval()
        predictor_loaded, predictor_path = load_predictor_weights(
            predictor, backbone_name, checkpoint_dir, trained_model_dir=trained_model_dir
        )
        feature_dim = int(predictor.feature_dim)
        policy_checkpoint_path = (
            trained_model_dir / POLICY_FILES[policy_name]
            if trained_model_dir is not None
            else None
        )
        policy, policy_loaded, policy_path = build_policy(
            policy_name,
            feature_dim,
            n_sensors,
            cfg.iql_hidden_dim,
            checkpoint_dir,
            load_checkpoint=trained_model_dir is not None or backbone_name == "lstm",
            checkpoint_path=policy_checkpoint_path,
        )
        policy = policy.cpu().eval()
        pipeline = DecisionPipeline(predictor, policy, n_active)

        predictor_params = count_parameters(predictor)
        policy_params = count_parameters(policy)
        train_cost = policy_training_cost(
            policy_name, feature_dim, n_sensors, policy.net[0].out_features
        )
        method_history = (
            read_history_path(trained_model_dir / "iql_history.json")
            if trained_model_dir is not None
            else histories[policy_name]
        )
        method_convergence = estimate_convergence(method_history)
        row = {
            "method": f"{backbone_name.upper()}+{policy_name.upper()}",
            "backbone": backbone_name,
            "policy": policy_name,
            "predictor_params": predictor_params,
            "policy_params": policy_params,
            "deployment_params": predictor_params + policy_params,
            **train_cost,
            "pipeline_training_resident_params": (
                predictor_params + train_cost["policy_training_resident_params"]
            ),
            "decision_flops": decision_flops(
                backbone_name, predictor, policy, window_size, n_sensors
            ),
            "deployment_weights_mib": module_tensor_bytes([predictor, policy]) / 1024 ** 2,
            "convergence_epoch": method_convergence["convergence_epoch"],
            "convergence_updates": method_convergence["convergence_updates"],
            "convergence_val_f1": method_convergence["convergence_val_f1"],
            "predictor_checkpoint_loaded": predictor_loaded,
            "policy_checkpoint_loaded": policy_loaded,
            "predictor_checkpoint": predictor_path,
            "policy_checkpoint": policy_path,
        }
        if not args.skip_latency:
            row.update(
                benchmark_latency(
                    pipeline, window_size, n_sensors, args.warmup, args.repeats
                )
            )
            row["p95_budget_utilization_pct"] = (
                100.0 * row["latency_p95_ms"] / max(args.decision_interval_ms, 1e-12)
            )
            row["p95_realtime_margin_x"] = (
                args.decision_interval_ms / max(row["latency_p95_ms"], 1e-12)
            )
        rows.append(row)
        print(
            f"[{row['method']}] params={row['deployment_params']:,} "
            f"FLOPs={row['decision_flops'] / 1e6:.2f}M "
            f"weights={row['deployment_weights_mib']:.2f} MiB"
            + (
                f" latency={row['latency_mean_ms']:.3f} ms "
                f"(p95={row['latency_p95_ms']:.3f})"
                if not args.skip_latency else ""
            )
        )

    csv_path = args.output_dir / "efficiency_summary.csv"
    json_path = args.output_dir / "efficiency_summary.json"
    write_csv(rows, csv_path)
    plot_outputs = [] if args.skip_plot else plot_learning_curves(histories, args.output_dir)
    payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cpu": cpu_description(),
            "torch_num_threads": torch.get_num_threads(),
            "window_size": window_size,
            "n_sensors": n_sensors,
            "n_active": n_active,
            "backbone_run_dir": str(backbone_run_dir) if backbone_run_dir else "",
            "transformer_config": {
                "d_model": compare_cfg.transformer_d_model,
                "nhead": compare_cfg.transformer_nhead,
                "ff_dim": compare_cfg.transformer_ff_dim,
                "layers": compare_cfg.transformer_layers,
                "dropout": compare_cfg.transformer_dropout,
            },
            "cnn_config": {
                "channels": list(compare_cfg.cnn_channels),
            },
            "patchtst_config": {
                "patch_len": compare_cfg.patch_len,
                "patch_stride": compare_cfg.patch_stride,
                "d_model": compare_cfg.patch_d_model,
                "nhead": compare_cfg.patch_nhead,
                "ff_dim": compare_cfg.patch_ff_dim,
                "layers": compare_cfg.patch_layers,
                "dropout": compare_cfg.patch_dropout,
            },
            "decision_interval_ms": float(args.decision_interval_ms),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "flop_definition": (
                "MAC=2 FLOPs; includes input masking, Linear, LSTM matrix products, "
                "Transformer QKV/attention/FFN, Conv1d, and policy Linear layers; "
                "excludes normalization, activations, memory movement, and Top-K comparisons."
            ),
            "latency_scope": (
                "CPU batch=1 raw history and mask through masking/concatenation, "
                "belief encoder, policy forward, and Top-M indices."
            ),
            "memory_scope": "Exact parameter and persistent/non-persistent buffer storage in MiB.",
            "parameter_scopes": {
                "deployment_params": "Predictor plus deployed policy network.",
                "policy_training_resident_params": (
                    "Policy-stage networks only: pi/Q/V for IQL, pi/online-Q/target-Q "
                    "for CQL, and pi for BC."
                ),
                "pipeline_training_resident_params": (
                    "Predictor plus all policy-stage resident networks; this is the "
                    "training parameter count directly comparable with deployment_params."
                ),
                "exclusions": "Optimizer states, gradients, activations, and offline data tensors.",
            },
            "validation_metric": (
                "Micro-F1 between policy Top-M masks and logged behavior masks on complete "
                "held-out trajectories; estimated Q is critic-based and method-specific."
            ),
            "convergence_definition": (
                "Epoch and cumulative optimizer updates at the maximum held-out action F1; "
                "this is a best-validation checkpoint proxy, not proof of critic convergence."
            ),
        },
        "rows": rows,
        "plot_outputs": plot_outputs,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"[Save] {csv_path}")
    print(f"[Save] {json_path}")


if __name__ == "__main__":
    main()
