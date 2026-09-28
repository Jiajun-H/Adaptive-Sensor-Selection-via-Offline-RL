"""
Helpers for artifact compatibility across changing data/label pipelines.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

import torch
import numpy as np


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _safe_path(path: Optional[str]) -> str:
    if not path:
        return ""
    return os.path.abspath(str(path))


def _experiment_ids(experiments) -> list:
    ids = []
    for idx, exp in enumerate(experiments or []):
        exp_id = ""
        if isinstance(exp, dict):
            exp_id = str(exp.get("exp_id", "")).strip()
        ids.append(exp_id or f"exp_{idx}")
    return ids


def build_artifact_signature(cfg, data_info: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    signature: Dict[str, Any] = {
        "signature_version": 4,
        "data_path": _safe_path(getattr(cfg, "data_path", "")),
        "data_dir": _safe_path(getattr(cfg, "data_dir", "")),
        "experiment_glob": str(getattr(cfg, "experiment_glob", "*.xlsx")),
        "experiment_max_files": int(getattr(cfg, "experiment_max_files", 0) or 0),
        "window_size": int(getattr(cfg, "window_size", 0)),
        "delta": int(getattr(cfg, "delta", 0)),
        "train_ratio": float(getattr(cfg, "train_ratio", 0.0)),
        "test_ratio": float(getattr(cfg, "test_ratio", 0.0)),
        "split_seed": int(getattr(cfg, "split_seed", 42)),
        "experiment_split_mode": str(getattr(cfg, "experiment_split_mode", "random")),
        "experiment_kfold_n_splits": int(getattr(cfg, "experiment_kfold_n_splits", 0) or 0),
        "experiment_kfold_index": int(getattr(cfg, "experiment_kfold_index", 0) or 0),
        "auto_reverse_experiments": bool(getattr(cfg, "auto_reverse_experiments", False)),
        "preprocessing_recipe": {
            "missing_values": "per_sensor_training_median",
            "normalization": "per_sensor_training_zscore",
            "fit_scope": "training_partition_only",
        },
        "model_recipe": {
            "hidden_dim": int(getattr(cfg, "hidden_dim", 0)),
            "num_layers": int(getattr(cfg, "num_layers", 0)),
            "lstm_dropout": float(getattr(cfg, "lstm_dropout", 0.0)),
            "head_dropout": float(getattr(cfg, "head_dropout", 0.0)),
        },
        "train_recipe": {
            "batch_size": int(getattr(cfg, "batch_size", 0)),
            "lr": float(getattr(cfg, "lr", 0.0)),
            "weight_decay": float(getattr(cfg, "weight_decay", 0.0)),
            "lambda_tte": float(getattr(cfg, "lambda_tte", 0.0)),
            "huber_delta": float(getattr(cfg, "huber_delta", 0.0)),
            "balanced_train_sampling": bool(getattr(cfg, "balanced_train_sampling", False)),
            "predictor_select_metric": str(getattr(cfg, "predictor_select_metric", "loss")),
        },
        "offline_recipe": {
            "iql_step_interval": int(getattr(cfg, "iql_step_interval", 0)),
            "iql_n_mask_samples": int(getattr(cfg, "iql_n_mask_samples", 0)),
            "iql_reward_lambda_cost": float(getattr(cfg, "iql_reward_lambda_cost", 0.0)),
            "reward_lambda_switch": float(getattr(cfg, "reward_lambda_switch", 0.0)),
            "reward_lambda_lead": float(getattr(cfg, "reward_lambda_lead", 0.0)),
            "reward_lambda_modality": float(getattr(cfg, "reward_lambda_modality", 0.0)),
            "reward_lambda_delta_risk": float(getattr(cfg, "reward_lambda_delta_risk", 0.0)),
            "behavior_policy_set": "random_static_dynamic_modality_mixture",
        },
        "rl_recipe": {
            "policy_loss": "balanced_action_bce",
            "iql_tau": float(getattr(cfg, "iql_tau", 0.0)),
            "iql_gamma": float(getattr(cfg, "iql_gamma", 0.0)),
            "iql_alpha": float(getattr(cfg, "iql_alpha", 0.0)),
            "iql_lr": float(getattr(cfg, "iql_lr", 0.0)),
            "iql_batch_size": int(getattr(cfg, "iql_batch_size", 0)),
            "iql_hidden_dim": int(getattr(cfg, "iql_hidden_dim", 0)),
            "iql_weight_clip": float(getattr(cfg, "iql_weight_clip", 0.0)),
            "cql_alpha": float(getattr(cfg, "cql_alpha", 0.0)),
            "cql_n_negative": int(getattr(cfg, "cql_n_negative", 0)),
            "cql_batch_size": int(getattr(cfg, "cql_batch_size", 0)),
            "cql_max_negative_eval_batch": int(getattr(cfg, "cql_max_negative_eval_batch", 0)),
            "cql_lr": float(getattr(cfg, "cql_lr", 0.0)),
            "bc_lr": float(getattr(cfg, "bc_lr", 0.0)),
            "soft_attn_lr": float(getattr(cfg, "soft_attn_lr", 0.0)),
            "soft_attn_batch_size": int(getattr(cfg, "soft_attn_batch_size", 0)),
        },
        "hazard_rule": {
            "temp_threshold": float(getattr(cfg, "temp_threshold", 0.0)),
            "trans_threshold": float(getattr(cfg, "trans_threshold", 0.0)),
            "heat_threshold": float(getattr(cfg, "heat_threshold", 0.0)),
            "co_threshold": float(getattr(cfg, "co_threshold", 0.0)),
            "hazard_min_modalities": int(getattr(cfg, "hazard_min_modalities", 1)),
            "hazard_min_consecutive": int(getattr(cfg, "hazard_min_consecutive", 1)),
        },
    }

    if data_info:
        signature["split_mode"] = str(data_info.get("split_mode", "unknown"))
        signature["n_sensors"] = int(data_info.get("n_sensors", getattr(cfg, "n_sensors", 0) or 0))
        if signature["split_mode"].startswith("experiment_"):
            signature["train_experiments"] = _experiment_ids(data_info.get("train_experiments", []))
            signature["val_experiments"] = _experiment_ids(data_info.get("val_experiments", []))
            signature["test_experiments"] = _experiment_ids(data_info.get("test_experiments", []))
        else:
            for key in ("train_range", "val_range", "test_range"):
                if key in data_info and data_info[key] is not None:
                    start, end = data_info[key]
                    signature[key] = [int(start), int(end)]
        if "preprocessing" in data_info:
            signature["preprocessing"] = _to_builtin(data_info["preprocessing"])
        if data_info.get("preprocessor") is not None:
            signature["preprocessor_stats"] = _to_builtin(data_info["preprocessor"])

    if extra:
        signature["extra"] = _to_builtin(extra)

    return _to_builtin(signature)


def artifact_signature_hash(signature: Optional[Dict[str, Any]]) -> str:
    if signature is None:
        return ""
    normalized = _to_builtin(signature)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def attach_artifact_signature(payload: Dict[str, Any], artifact_signature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(payload)
    if artifact_signature is not None:
        normalized = _to_builtin(artifact_signature)
        out["artifact_signature"] = normalized
        out["artifact_signature_hash"] = artifact_signature_hash(normalized)
    return out


def artifact_matches(
    artifact_path: str,
    n_sensors: Optional[int] = None,
    artifact_signature: Optional[Dict[str, Any]] = None,
) -> bool:
    if not os.path.exists(artifact_path):
        return False

    try:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except Exception:
        return False

    if not isinstance(artifact, dict):
        return False

    if n_sensors is not None:
        stored_n = artifact.get("n_sensors")
        if stored_n is not None:
            try:
                if int(stored_n) != int(n_sensors):
                    return False
            except Exception:
                return False

    if artifact_signature is None:
        return True

    expected_hash = artifact_signature_hash(artifact_signature)
    stored_hash = artifact.get("artifact_signature_hash")
    if stored_hash is not None:
        return str(stored_hash) == expected_hash

    stored_signature = artifact.get("artifact_signature")
    if stored_signature is None:
        return False
    return artifact_signature_hash(stored_signature) == expected_hash
