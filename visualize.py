"""
visualize.py - plotting helpers.

Provides:
  1. plot_pareto_curves()
  2. plot_timeline_warning()
  3. plot_sensor_heatmap()
  4. plot_ablation_bar_chart()
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _first_positive_idx(arr: np.ndarray) -> int:
    idx = np.where(np.asarray(arr) > 0.5)[0]
    return int(idx[0]) if len(idx) > 0 else int(len(arr))


def _clip_onset(onset: int, length: int) -> int:
    return max(0, min(int(onset), max(0, int(length))))


# ======================================================================
# 1. Pareto curves
# ======================================================================


def plot_pareto_curves(pareto_results: dict, save_dir: str = "figures"):
    """
    Plot Pareto curves: AUC/F1 and lead time against active sensor count M.

    Args:
        pareto_results: {method_name: {m_value: metrics_dict, ...}, ...}
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(pareto_results)))
    markers = ["o", "s", "^", "D", "v", "p", "*"]

    for idx, (method, m_results) in enumerate(pareto_results.items()):
        m_vals = sorted(m_results.keys())
        aucs = [m_results[m]["auc"] for m in m_vals]
        f1s = [m_results[m]["f1"] for m in m_vals]
        leads = [m_results[m]["lead_time"]["avg_lead"] for m in m_vals]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        axes[0].plot(
            m_vals, aucs, marker=marker, color=color, label=method, linewidth=2, markersize=8
        )
        axes[1].plot(
            m_vals, f1s, marker=marker, color=color, label=method, linewidth=2, markersize=8
        )
        axes[2].plot(
            m_vals, leads, marker=marker, color=color, label=method, linewidth=2, markersize=8
        )

    axes[0].set_xlabel("Active Sensors M", fontsize=12)
    axes[0].set_ylabel("AUC", fontsize=12)
    axes[0].set_title("AUC vs M", fontsize=13)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Active Sensors M", fontsize=12)
    axes[1].set_ylabel("F1 Score", fontsize=12)
    axes[1].set_title("F1 vs M", fontsize=13)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Active Sensors M", fontsize=12)
    axes[2].set_ylabel("Avg Lead Time (s)", fontsize=12)
    axes[2].set_title("Lead Time vs M", fontsize=13)
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "pareto_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Pareto curves saved to: {path}")


# ======================================================================
# 2. Timeline warning plot
# ======================================================================


def plot_timeline_warning(
    method_probs: dict,
    cls_labels: np.ndarray,
    fire_onset: int,
    window_size: int = 60,
    threshold: float = 0.5,
    save_dir: str = "figures",
    experiment_infos=None,
):
    """
    Plot a single timeline chart for risk probabilities.

    In multi-experiment mode, pass experiment_infos so the total chart keeps one
    global x-axis while danger zones are reset for each experiment.

    Args:
        method_probs: {method_name: probs_array}
        cls_labels: full labels on the plotting axis
        fire_onset: used in single-sequence mode
        window_size: model window size
        threshold: optional horizontal alarm threshold
        save_dir: output directory
        experiment_infos: optional list of dicts with:
            {"length": int, "warning_onset": int, "fire_onset": int, "exp_id": str(optional)}
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    use_segments = experiment_infos is not None and len(experiment_infos) > 0
    warning_zone_color = "#ffcc80"
    danger_zone_color = "#ef5350"
    if use_segments:
        total_steps = int(sum(max(0, int(exp.get("length", 0))) for exp in experiment_infos))
        offset = 0
        warning_label_used = False
        danger_label_used = False
        for idx, exp in enumerate(experiment_infos):
            exp_len = max(0, int(exp.get("length", 0)))
            exp_warning_onset = _clip_onset(exp.get("warning_onset", exp_len), exp_len)
            exp_fire_onset = _clip_onset(exp.get("fire_onset", exp_len), exp_len)

            if idx > 0:
                ax.axvline(x=offset, color="black", linestyle=":", linewidth=0.8, alpha=0.35)

            if 0 <= exp_warning_onset < exp_len:
                ax.axvspan(
                    offset + exp_warning_onset,
                    offset + exp_len,
                    alpha=0.18,
                    color=warning_zone_color,
                    label="Warning Zone" if not warning_label_used else None,
                )
                warning_label_used = True

            if 0 <= exp_fire_onset < exp_len:
                ax.axvspan(
                    offset + exp_fire_onset,
                    offset + exp_len,
                    alpha=0.22,
                    color=danger_zone_color,
                    label="Danger Zone" if not danger_label_used else None,
                )
                ax.axvline(
                    x=offset + exp_fire_onset,
                    color="darkred",
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.55,
                )
                danger_label_used = True

            offset += exp_len
    else:
        total_steps = len(cls_labels)
        warning_onset = _first_positive_idx(cls_labels)
        fire_onset = _clip_onset(fire_onset, total_steps)
        if 0 <= warning_onset < total_steps:
            ax.axvspan(
                warning_onset,
                total_steps,
                alpha=0.18,
                color=warning_zone_color,
                label="Warning Zone",
            )
        if 0 <= fire_onset < total_steps:
            ax.axvspan(
                fire_onset,
                total_steps,
                alpha=0.22,
                color=danger_zone_color,
                label="Danger Zone",
            )

    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="gray",
            linestyle="--",
            alpha=0.6,
            label=f"Threshold={threshold:.3f}",
        )

    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (method, probs) in enumerate(method_probs.items()):
        color = colors_list[idx % len(colors_list)]
        if use_segments:
            prob_offset = 0
            time_offset = 0
            labeled = False
            for exp in experiment_infos:
                exp_len = max(0, int(exp.get("length", 0)))
                eval_len = max(0, exp_len - window_size + 1)
                local_probs = probs[prob_offset:prob_offset + eval_len]
                if len(local_probs) > 0:
                    t_axis = np.arange(
                        time_offset + window_size - 1,
                        time_offset + window_size - 1 + len(local_probs),
                    )
                    ax.plot(
                        t_axis,
                        local_probs,
                        color=color,
                        label=method if not labeled else None,
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    labeled = True
                prob_offset += eval_len
                time_offset += exp_len
        else:
            t_start = window_size - 1
            t_axis = np.arange(t_start, t_start + len(probs))
            ax.plot(t_axis, probs, color=color, label=method, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Risk Probability", fontsize=12)
    ax.set_title("Timeline Warning Probability", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    if total_steps > 0:
        ax.set_xlim(0, total_steps - 1)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    if not use_segments:
        ax.axvline(x=fire_onset, color="darkred", linestyle="-", linewidth=1.5, alpha=0.7)
        ax.text(
            fire_onset + 5,
            0.95,
            f"fire onset={fire_onset}",
            fontsize=9,
            color="darkred",
            verticalalignment="top",
        )

    plt.tight_layout()
    path = os.path.join(save_dir, "timeline_warning.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Timeline warning plot saved to: {path}")


# ======================================================================
# 3. Sensor selection heatmap
# ======================================================================


def plot_sensor_heatmap(
    masks_over_time: np.ndarray,
    method_name: str = "IQL",
    sensor_names=None,
    fire_onset=None,
    window_size: int = 60,
    save_dir: str = "figures",
    top_k: int = 30,
):
    """
    Plot sensor selection heatmap (time x sensors).

    Args:
        masks_over_time: (T_eval, N) binary mask matrix
        method_name: title suffix
        sensor_names: optional sensor names
        fire_onset: optional fire onset marker
        window_size: window size
        save_dir: output directory
        top_k: display top-k most selected sensors
    """
    os.makedirs(save_dir, exist_ok=True)

    T_eval, N = masks_over_time.shape

    freq = masks_over_time.sum(axis=0)
    top_indices = np.argsort(freq)[::-1][:top_k]
    sub_masks = masks_over_time[:, top_indices].T

    if sensor_names is not None:
        y_labels = [sensor_names[i] if i < len(sensor_names) else f"S{i}" for i in top_indices]
    else:
        y_labels = [f"S{i}" for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, max(6, top_k * 0.25)))

    im = ax.imshow(sub_masks, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=1)

    if fire_onset is not None:
        t_onset_in_eval = fire_onset - (window_size - 1)
        if 0 < t_onset_in_eval < T_eval:
            ax.axvline(
                x=t_onset_in_eval,
                color="blue",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Fire Onset",
            )

    ax.set_xlabel("Time Step (from W-1)", fontsize=11)
    ax.set_ylabel("Sensor", fontsize=11)
    ax.set_title(f"Sensor Selection Heatmap - {method_name}", fontsize=13)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))

    plt.colorbar(im, ax=ax, label="Selected", shrink=0.6)
    if fire_onset is not None:
        ax.legend(fontsize=9)

    plt.tight_layout()
    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(save_dir, f"sensor_heatmap_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Sensor heatmap saved to: {path}")


# ======================================================================
# 4. Ablation bar chart
# ======================================================================


def plot_ablation_bar_chart(ablation_results: dict, save_dir: str = "figures"):
    """
    Plot bar chart for ablation results.

    Args:
        ablation_results: {config_name: metrics_dict, ...}
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    configs = list(ablation_results.keys())
    f1_vals = [ablation_results[c]["f1"] for c in configs]
    auc_vals = [ablation_results[c]["auc"] for c in configs]
    lead_vals = [ablation_results[c]["lead_time"]["avg_lead"] for c in configs]

    x = np.arange(len(configs))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bars1 = axes[0].bar(x, f1_vals, width, color="#2196F3", edgecolor="white")
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("Ablation: F1 Score", fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylim(0, max(f1_vals) * 1.15 if max(f1_vals) > 0 else 1.0)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, f1_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars2 = axes[1].bar(x, auc_vals, width, color="#4CAF50", edgecolor="white")
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].set_title("Ablation: AUC", fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylim(0, max(auc_vals) * 1.15 if max(auc_vals) > 0 else 1.0)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, auc_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars3 = axes[2].bar(x, lead_vals, width, color="#FF9800", edgecolor="white")
    axes[2].set_ylabel("Avg Lead Time (s)", fontsize=12)
    axes[2].set_title("Ablation: Lead Time", fontsize=13)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    axes[2].set_ylim(0, max(lead_vals) * 1.25 if max(lead_vals) > 0 else 60.0)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, lead_vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Ablation bar chart saved to: {path}")


# ======================================================================
if __name__ == "__main__":
    sanity_dir = os.path.join("figures", "sanity")

    print("Testing plot_pareto_curves...")
    dummy_pareto = {
        "IQL": {
            5: {"auc": 0.85, "f1": 0.80, "lead_time": {"avg_lead": 50}},
            10: {"auc": 0.90, "f1": 0.88, "lead_time": {"avg_lead": 55}},
            20: {"auc": 0.95, "f1": 0.93, "lead_time": {"avg_lead": 60}},
        },
        "CQL": {
            5: {"auc": 0.82, "f1": 0.78, "lead_time": {"avg_lead": 45}},
            10: {"auc": 0.88, "f1": 0.85, "lead_time": {"avg_lead": 52}},
            20: {"auc": 0.93, "f1": 0.91, "lead_time": {"avg_lead": 58}},
        },
    }
    plot_pareto_curves(dummy_pareto, sanity_dir)

    print("Testing plot_timeline_warning...")
    T = 500
    probs1 = np.random.rand(T - 59)
    probs2 = np.random.rand(T - 59)
    cls = np.zeros(T)
    cls[200:] = 1
    plot_timeline_warning({"IQL": probs1, "CQL": probs2}, cls, fire_onset=200, save_dir=sanity_dir)

    print("Testing plot_sensor_heatmap...")
    masks = np.random.binomial(1, 0.1, (T - 59, 179))
    plot_sensor_heatmap(masks, "IQL", fire_onset=200, save_dir=sanity_dir)

    print("visualize.py sanity check passed.")
