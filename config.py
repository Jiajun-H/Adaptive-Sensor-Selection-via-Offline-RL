"""
Central config for risk predictor + offline RL policies.
"""

from dataclasses import dataclass
from typing import List
import os


@dataclass
class Config:
    # ---- paths ----
    data_path: str = os.path.join(os.path.dirname(__file__), "data.xlsx")
    data_dir: str = os.path.join(os.path.dirname(__file__), "converted_excel")
    experiment_glob: str = "*.xlsx"
    experiment_max_files: int = 0
    save_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")
    fig_dir: str = os.path.join(os.path.dirname(__file__), "figures")

    # ---- data ----
    window_size: int = 60
    delta: int = 30
    val_ratio: float = 0.2
    test_ratio: float = 0.15
    train_ratio: float = 0.70
    split_seed: int = 42
    experiment_split_mode: str = "random"  # random | kfold
    experiment_kfold_n_splits: int = 5
    experiment_kfold_index: int = 0

    # ---- sensors ----
    n_sensors: int = 0
    n_active: int = 20

    # ---- hazard thresholds ----
    temp_threshold: float = 200.0
    trans_threshold: float = 0.3
    heat_threshold: float = 2.5
    co_threshold: float = 100.0
    hazard_min_modalities: int = 2
    hazard_min_consecutive: int = 3
    # Direction must come from external acquisition metadata. Inferring it from
    # the full sequence would expose future observations during evaluation.
    auto_reverse_experiments: bool = False

    # ---- predictor ----
    hidden_dim: int = 128
    num_layers: int = 2
    lstm_dropout: float = 0.2
    head_dropout: float = 0.3

    # ---- stage-1 train ----
    batch_size: int = 256
    data_num_workers: int = 8
    data_prefetch_factor: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_tte: float = 0.5
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    patience: int = 15
    balanced_train_sampling: bool = True
    min_pos_weight: float = 0.1
    max_pos_weight: float = 30.0
    predictor_select_metric: str = "f1_tuned"

    # ---- evaluation ----
    alarm_threshold: float = 0.5
    tune_alarm_threshold: bool = True
    threshold_metric: str = "f1"
    threshold_search_min: float = 0.05
    threshold_search_max: float = 0.95
    threshold_search_steps: int = 19
    policy_belief_mode: str = "full"  # full | previous_mask

    # ---- IQL ----
    iql_tau: float = 0.8
    iql_gamma: float = 0.99
    iql_alpha: float = 0.3
    iql_lr: float = 3e-4
    iql_epochs: int = 200
    iql_batch_size: int = 8192
    rl_num_workers: int = 0
    rl_prefetch_factor: int = 4
    rl_preload_to_device: bool = True
    rl_val_ratio: float = 0.1
    rl_val_seed: int = 2026
    rl_val_max_samples: int = 8192
    iql_hidden_dim: int = 256
    iql_weight_clip: float = 100.0
    iql_step_interval: int = 5
    iql_n_mask_samples: int = 15
    iql_reward_lambda_cost: float = 0.1
    reward_lambda_modality: float = 0.15
    reward_lambda_delta_risk: float = 0.05
    iql_grad_clip: float = 1.0
    iql_patience: int = 30
    offline_build_batch_steps: int = 64
    offline_encode_batch_size: int = 4096

    # ---- CQL ----
    cql_alpha: float = 1.0
    cql_n_negative: int = 20
    cql_batch_size: int = 8192
    cql_max_negative_eval_batch: int = 65536
    cql_epochs: int = 200
    cql_lr: float = 3e-4
    cql_target_ema: float = 0.005

    # ---- soft attention ----
    soft_attn_epochs: int = 100
    soft_attn_lr: float = 3e-4
    soft_attn_batch_size: int = 1024
    soft_attn_embed_batch_size: int = 2048
    soft_attn_preload_to_device: bool = True

    # ---- reward shaping ----
    reward_lambda_switch: float = 0.05
    reward_lambda_lead: float = 0.2

    # ---- BC ----
    bc_epochs: int = 100
    bc_lr: float = 3e-4

    # ---- pareto ----
    pareto_m_values: List[int] = None  # type: ignore

    # ---- runtime ----
    device: str = "cuda"
    offline_device: str = "cuda"
    use_amp: bool = False
    allow_tf32: bool = True
    policy_eval_batch_size: int = 2048

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        if self.pareto_m_values is None:
            self.pareto_m_values = [5, 10, 15, 20, 30]
