"""Excel loading and sensor-label helpers used only by Chapter 5 figures.

This explanatory-figure path interpolates within each workbook and can reverse
an experiment from its signal trend. Training data preparation remains in
dataset.py, which fits imputation and normalization on the training split.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FAMILY_ORDER = ["temperature", "co", "transmittance", "heat_flux", "smoke_layer", "unknown"]
FAMILY_DIRECTION = {
    "temperature": 1.0,
    "co": 1.0,
    "heat_flux": 1.0,
    "smoke_layer": 1.0,
    "transmittance": -1.0,
    "unknown": 1.0,
}
TEMP_PROBE_RE = re.compile(r"^\d+-\d+(测点温度)?$")
HEAT_PROBE_RE = re.compile(r"^\d+(\.\d+)?d$")


@dataclass
class ExperimentData:
    path: Path
    exp_id: str
    data: np.ndarray
    cats: list[str]
    names: list[str]
    sensor_types: list[str]
    labels: np.ndarray
    ttes: np.ndarray
    modality_hits: np.ndarray
    fire_onset: int
    warning_onset: int

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text

def _normalize_header_text(value: Any) -> str:
    text = _safe_text(value).lower()
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
        text = text.replace(old, new)
    return text

def infer_sensor_type(cat: str, name: str) -> str:
    cat_n = _normalize_header_text(cat)
    name_n = _normalize_header_text(name)
    text = f"{cat_n}|{name_n}"
    if (
        "temperature" in text
        or "temp" in text
        or "温度" in text
        or "℃" in str(cat)
        or "℃" in str(name)
        or TEMP_PROBE_RE.fullmatch(name_n) is not None
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
        or HEAT_PROBE_RE.fullmatch(name_n) is not None
    ):
        return "heat_flux"
    if "烟气层" in text or "smokelayer" in text or "smoke" in text:
        return "smoke_layer"
    if "co浓度" in text or cat_n.startswith("co") or name_n == "co":
        return "co"
    return "unknown"

def infer_sensor_types(cats: list[str], names: list[str]) -> list[str]:
    return [infer_sensor_type(cats[i], names[i]) for i in range(len(names))]

def group_indices(sensor_types: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, sensor_type in enumerate(sensor_types):
        groups.setdefault(sensor_type, []).append(idx)
    return groups

def family_series(data: np.ndarray, indices: list[int], sensor_type: str) -> np.ndarray:
    if not indices:
        return np.array([], dtype=np.float32)
    values = data[:, indices]
    if sensor_type == "transmittance":
        return np.nanmin(values, axis=1).astype(np.float32)
    return np.nanmax(values, axis=1).astype(np.float32)

def first_sustained_true(mask: np.ndarray, min_consecutive: int) -> int | None:
    run_len = 0
    start = 0
    for idx, is_true in enumerate(mask):
        if bool(is_true):
            if run_len == 0:
                start = idx
            run_len += 1
            if run_len >= min_consecutive:
                return start
        else:
            run_len = 0
    return None

def generate_labels(
    data: np.ndarray,
    cats: list[str],
    names: list[str],
    temp_threshold: float,
    trans_threshold: float,
    heat_threshold: float,
    co_threshold: float,
    min_modalities: int,
    min_consecutive: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor_types = infer_sensor_types(cats, names)
    groups = group_indices(sensor_types)
    flags: list[np.ndarray] = []
    if groups.get("temperature"):
        flags.append(family_series(data, groups["temperature"], "temperature") > temp_threshold)
    if groups.get("transmittance"):
        flags.append(family_series(data, groups["transmittance"], "transmittance") < trans_threshold)
    if groups.get("heat_flux"):
        flags.append(family_series(data, groups["heat_flux"], "heat_flux") > heat_threshold)
    if groups.get("co"):
        flags.append(family_series(data, groups["co"], "co") > co_threshold)

    t_steps = data.shape[0]
    labels = np.zeros(t_steps, dtype=np.float32)
    ttes = np.full(t_steps, float(t_steps + 1), dtype=np.float32)
    if not flags:
        return labels, ttes, np.zeros(t_steps, dtype=np.int32)

    modality_hits = np.zeros(t_steps, dtype=np.int32)
    for flag in flags:
        modality_hits += flag.astype(np.int32)

    required = min(max(int(min_modalities), 1), len(flags))
    onset = first_sustained_true(modality_hits >= required, min_consecutive)
    if onset is None:
        return labels, ttes, modality_hits

    labels[onset:] = 1.0
    ttes[onset:] = 0.0
    if onset > 0:
        ttes[:onset] = np.arange(onset, 0, -1, dtype=np.float32)
    return labels, ttes, modality_hits

def trend_vote(series: np.ndarray, danger_increases: bool, base_margin: float) -> int:
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

def should_reverse_experiment(data: np.ndarray, cats: list[str], names: list[str]) -> bool:
    sensor_types = infer_sensor_types(cats, names)
    groups = group_indices(sensor_types)
    votes: list[int] = []
    checks = [
        ("temperature", True, 5.0),
        ("co", True, 20.0),
        ("heat_flux", True, 0.2),
        ("transmittance", False, 0.05),
    ]
    for sensor_type, danger_increases, base_margin in checks:
        indices = groups.get(sensor_type, [])
        if not indices:
            continue
        vote = trend_vote(
            family_series(data, indices, sensor_type),
            danger_increases=danger_increases,
            base_margin=base_margin,
        )
        if vote != 0:
            votes.append(vote)
    if not votes:
        return False
    forward_votes = sum(1 for vote in votes if vote > 0)
    reverse_votes = sum(1 for vote in votes if vote < 0)
    return reverse_votes >= 2 and reverse_votes > forward_votes

def load_excel_experiment(
    path: Path,
    *,
    broken_val: float,
    broken_ratio: float,
    auto_reverse: bool,
    delta: int,
    fire_onset_override: int | None,
    temp_threshold: float,
    trans_threshold: float,
    heat_threshold: float,
    co_threshold: float,
    min_modalities: int,
    min_consecutive: int,
) -> ExperimentData:
    df_raw = pd.read_excel(path, header=None, engine="openpyxl")
    if len(df_raw) < 3:
        raise ValueError(f"{path} has fewer than three rows.")

    row_cat = df_raw.iloc[0].replace("", np.nan).ffill()
    row_name = df_raw.iloc[1]
    df_data = df_raw.iloc[2:].reset_index(drop=True)
    skip_exact = {"时间", "TTime", "设备地址", "气体名称", "单位", "报警状态", "通道"}

    cols: list[np.ndarray] = []
    cats: list[str] = []
    names: list[str] = []
    for col_idx in range(df_data.shape[1]):
        series = pd.to_numeric(df_data.iloc[:, col_idx], errors="coerce").replace(broken_val, np.nan)
        if int(series.notna().sum()) < max(3, int(len(series) * broken_ratio)):
            continue

        name = _safe_text(row_name.iloc[col_idx] if col_idx < len(row_name) else "")
        cat = _safe_text(row_cat.iloc[col_idx] if col_idx < len(row_cat) else "")
        if not name:
            name = f"col_{col_idx}"
        if name in skip_exact or cat in skip_exact:
            continue

        filled = series.interpolate(limit_direction="both").ffill().bfill()
        if filled.notna().sum() == 0:
            continue
        arr = filled.to_numpy(dtype=np.float32)
        if np.nanstd(arr) < 1e-12 and np.nanmedian(arr) == broken_val:
            continue

        cols.append(arr)
        cats.append(cat)
        names.append(name)

    if not cols:
        raise ValueError(f"No usable numeric sensor columns found in {path}.")

    data = np.stack(cols, axis=1).astype(np.float32)
    if auto_reverse and should_reverse_experiment(data, cats, names):
        data = data[::-1].copy()

    labels, ttes, modality_hits = generate_labels(
        data,
        cats,
        names,
        temp_threshold=temp_threshold,
        trans_threshold=trans_threshold,
        heat_threshold=heat_threshold,
        co_threshold=co_threshold,
        min_modalities=min_modalities,
        min_consecutive=min_consecutive,
    )
    fire_onset = int(np.where(labels > 0.5)[0][0]) if np.any(labels > 0.5) else len(labels)
    if fire_onset_override is not None:
        fire_onset = max(0, min(int(fire_onset_override), len(labels)))
        labels = np.zeros(len(labels), dtype=np.float32)
        ttes = np.full(len(labels), float(len(labels) + 1), dtype=np.float32)
        if fire_onset < len(labels):
            labels[fire_onset:] = 1.0
            ttes[fire_onset:] = 0.0
            if fire_onset > 0:
                ttes[:fire_onset] = np.arange(fire_onset, 0, -1, dtype=np.float32)
    warning_onset = max(0, fire_onset - int(delta)) if fire_onset < len(labels) else len(labels)

    return ExperimentData(
        path=path,
        exp_id=path.stem,
        data=data,
        cats=cats,
        names=names,
        sensor_types=infer_sensor_types(cats, names),
        labels=labels,
        ttes=ttes,
        modality_hits=modality_hits,
        fire_onset=fire_onset,
        warning_onset=warning_onset,
    )

def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.zeros_like(values, dtype=np.float32)
    center = float(np.median(valid))
    q25, q75 = np.quantile(valid, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale < 1e-8:
        scale = float(np.std(valid))
    if scale < 1e-8:
        scale = 1.0
    return ((values - center) / scale).astype(np.float32)
