"""
Baseline models and training helpers.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np

from artifact_utils import attach_artifact_signature


def _tensor_nbytes(tensors) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def _maybe_preload_tensors_to_device(tensors, device, enabled: bool, prefix: str):
    if device.type != "cuda" or not enabled:
        return tensors
    moved = []
    try:
        for tensor in tensors:
            moved.append(tensor.to(device, non_blocking=True))
        print(f"[{prefix}] preloaded tensors to {device} ({_tensor_nbytes(moved) / 1024 ** 2:.1f} MiB)")
        return tuple(moved)
    except RuntimeError as exc:
        print(f"[{prefix}] preload to {device} failed, using CPU tensors: {exc}")
        del moved
        torch.cuda.empty_cache()
        return tensors


def _build_soft_attention_tensors(exp_arrays, valid_indices, window: int, n_sensors: int):
    n = len(valid_indices)
    obs = np.empty((n, window, n_sensors), dtype=np.float32)
    labels = np.empty(n, dtype=np.float32)
    ttes_out = np.empty(n, dtype=np.float32)
    for out_idx, (exp_idx, t) in enumerate(valid_indices):
        exp = exp_arrays[exp_idx]
        obs[out_idx] = exp["data_norm"][t - window + 1:t + 1]
        labels[out_idx] = exp["cls_labels"][t]
        ttes_out[out_idx] = exp["ttes"][t]
    return (
        torch.from_numpy(obs),
        torch.from_numpy(labels),
        torch.from_numpy(ttes_out),
    )


@torch.no_grad()
def _precompute_full_beliefs(predictor, obs_all: torch.Tensor, n_sensors: int, batch_size: int, device):
    predictor.eval()
    h_chunks = []
    batch_size = max(1, int(batch_size))
    for start in range(0, obs_all.shape[0], batch_size):
        obs = obs_all[start:start + batch_size].to(device, non_blocking=True)
        mask_full = torch.ones(
            obs.shape[0],
            obs.shape[1],
            n_sensors,
            dtype=obs.dtype,
            device=device,
        )
        x_full = torch.cat([obs, mask_full], dim=-1)
        h_chunks.append(predictor.encode_belief(x_full).detach().cpu())
    return torch.cat(h_chunks, dim=0) if h_chunks else torch.empty(0, 0)


class LogisticRegressionPredictor(nn.Module):
    """
    A lightweight predictor with the same interface as MaskAwareRiskPredictor:
    - forward(x) -> (logit, tte_hat)
    - encode_belief(x) -> feature
    - predict(x) -> (prob, tte_hat)
    """

    def __init__(self, n_sensors: int, hidden_dim: int = 128):
        super().__init__()
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim
        stat_dim = 3 * n_sensors
        self.feature_proj = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def _extract_stats(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n_sensors
        obs = x[:, -10:, :n]
        mask = x[:, -1, n:]
        obs_mean = obs.mean(dim=1)
        obs_std = obs.std(dim=1)
        return torch.cat([obs_mean, obs_std, mask], dim=-1)

    def forward(self, x: torch.Tensor):
        stats = self._extract_stats(x)
        feat = self.feature_proj(stats)
        logit = self.cls_head(feat).squeeze(-1)
        tte_hat = self.tte_head(feat).squeeze(-1)
        return logit, tte_hat

    def encode_belief(self, x: torch.Tensor) -> torch.Tensor:
        stats = self._extract_stats(x)
        return self.feature_proj(stats)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        logit, tte_hat = self.forward(x)
        p_hat = torch.sigmoid(logit)
        return p_hat, tte_hat


class SoftAttentionGating(nn.Module):
    """
    Soft gating network for sensor selection.
    """

    def __init__(self, hidden_dim: int, n_sensors: int, mlp_dim: int = 256):
        super().__init__()
        self.n_sensors = n_sensors
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, n_sensors),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h))

    def forward_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

    @torch.no_grad()
    def select_top_m(self, h: torch.Tensor, m: int) -> torch.Tensor:
        self.eval()
        logits = self.net(h)
        _, topk_idx = logits.topk(m, dim=-1)
        mask = torch.zeros_like(logits)
        mask.scatter_(1, topk_idx, 1.0)
        return mask


def train_logistic_predictor(cfg, train_loader, val_loader, n_sensors, device, artifact_signature=None):
    from evaluate import evaluate
    from model import MultiTaskLoss

    model = LogisticRegressionPredictor(n_sensors, cfg.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[LogisticPredictor] params: {n_params:,}")

    n_pos, n_neg = 0, 0
    for i in range(len(train_loader.dataset)):
        _, l, _ = train_loader.dataset[i]
        if l > 0.5:
            n_pos += 1
        else:
            n_neg += 1
    pos_weight = min(max(n_neg / max(n_pos, 1), 0.1), 10.0)

    criterion = MultiTaskLoss(
        lambda_tte=cfg.lambda_tte,
        huber_delta=cfg.huber_delta,
        pos_weight=pos_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    epochs = cfg.epochs
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(cfg.save_dir, "logistic_predictor.pt")
    print(f"[LogisticPredictor] start training for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, label, tte_true in train_loader:
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            tte_true = tte_true.to(device, non_blocking=True)
            logit, tte_hat = model(x)
            loss, _, _ = criterion(logit, tte_hat, label, tte_true)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(n_batches, 1)

        val_metrics = evaluate(model, criterion, val_loader, str(device))
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train={train_loss:.4f} Val={val_metrics['loss']:.4f} "
                f"F1={val_metrics.get('f1', 0):.3f}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                attach_artifact_signature(
                    {
                        "model_state_dict": model.state_dict(),
                        "n_sensors": n_sensors,
                        "val_metrics": val_metrics,
                    },
                    artifact_signature,
                ),
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stop @ epoch {epoch}")
                break

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[LogisticPredictor] done, best val_loss={best_val_loss:.4f}")
    return model


def train_soft_attention(
    cfg,
    predictor,
    data_norm: np.ndarray,
    cls_labels: np.ndarray,
    ttes: np.ndarray,
    train_range: tuple,
    n_sensors: int,
    device,
    train_experiments=None,
    artifact_signature=None,
):
    from model import MultiTaskLoss

    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    w = cfg.window_size
    gating = SoftAttentionGating(cfg.hidden_dim, n_sensors, cfg.iql_hidden_dim).to(device)
    n_params = sum(p.numel() for p in gating.parameters())
    print(f"[SoftAttention] params: {n_params:,}")

    optimizer = torch.optim.Adam(gating.parameters(), lr=cfg.soft_attn_lr)
    epochs = cfg.soft_attn_epochs

    if train_experiments is not None and len(train_experiments) > 0:
        train_cls = np.concatenate([exp["cls_labels"] for exp in train_experiments], axis=0)
    else:
        t0, t1 = train_range
        train_cls = cls_labels[t0:t1]
    n_pos = int((train_cls == 1).sum())
    n_neg = int((train_cls == 0).sum())
    pos_weight = min(max(n_neg / max(n_pos, 1), 0.1), 10.0)

    criterion = MultiTaskLoss(
        lambda_tte=cfg.lambda_tte,
        huber_delta=cfg.huber_delta,
        pos_weight=pos_weight,
    ).to(device)

    if train_experiments is not None and len(train_experiments) > 0:
        exp_arrays = [
            {
                "data_norm": exp["data_norm"],
                "cls_labels": exp["cls_labels"],
                "ttes": exp["ttes"],
            }
            for exp in train_experiments
        ]
        valid_indices = []
        for exp_idx, exp in enumerate(exp_arrays):
            t_steps = len(exp["data_norm"])
            valid_indices.extend([(exp_idx, t) for t in range(w - 1, t_steps)])
    else:
        t0, t1 = train_range
        exp_arrays = [{"data_norm": data_norm, "cls_labels": cls_labels, "ttes": ttes}]
        valid_indices = [(0, t) for t in range(max(t0, w - 1), t1)]

    batch_size = max(1, int(getattr(cfg, "soft_attn_batch_size", getattr(cfg, "batch_size", 128))))
    embed_batch_size = max(1, int(getattr(cfg, "soft_attn_embed_batch_size", 2048)))
    save_path = os.path.join(cfg.save_dir, "soft_attention.pt")
    best_loss = float("inf")
    patience_counter = 0
    l1_lambda = 0.01
    print(
        f"[SoftAttention] start training {epochs} epochs, windows={len(valid_indices)}, "
        f"batch_size={batch_size}"
    )

    obs_all, labels_all, ttes_all = _build_soft_attention_tensors(
        exp_arrays,
        valid_indices,
        w,
        n_sensors,
    )
    h_full_all = _precompute_full_beliefs(
        predictor,
        obs_all,
        n_sensors,
        embed_batch_size,
        device,
    )
    obs_all, labels_all, ttes_all, h_full_all = _maybe_preload_tensors_to_device(
        (obs_all, labels_all, ttes_all, h_full_all),
        device,
        bool(getattr(cfg, "soft_attn_preload_to_device", True)),
        "SoftAttention",
    )
    n_windows = int(obs_all.shape[0])

    predictor_was_training = predictor.training
    rnn_states = []
    for module in predictor.modules():
        if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
            rnn_states.append((module, module.training, module.dropout))
            module.train()
            module.dropout = 0.0

    try:
        for epoch in range(1, epochs + 1):
            gating.train()
            if getattr(obs_all, "device", torch.device("cpu")).type == "cuda":
                order = torch.randperm(n_windows, device=device)
            else:
                order = torch.from_numpy(np.random.permutation(n_windows))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_windows, batch_size):
                idx = order[start:start + batch_size]
                if idx.numel() < 2:
                    continue

                obs_t = obs_all[idx].to(device, non_blocking=True)
                labels_t = labels_all[idx].to(device, non_blocking=True)
                ttes_t = ttes_all[idx].to(device, non_blocking=True)
                h = h_full_all[idx].to(device, non_blocking=True)

                soft_weights = gating(h)
                obs_masked = obs_t * soft_weights.unsqueeze(1)
                mask_soft_exp = soft_weights.unsqueeze(1).expand(-1, w, -1)
                x_gated = torch.cat([obs_masked, mask_soft_exp], dim=-1)

                logit, tte_hat = predictor(x_gated)
                loss, _, _ = criterion(logit, tte_hat, labels_t, ttes_t)
                sparsity_loss = l1_lambda * soft_weights.mean()
                total_loss = loss + sparsity_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(gating.parameters(), cfg.iql_grad_clip)
                optimizer.step()

                epoch_loss += float(total_loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | Loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(
                    attach_artifact_signature(
                        {
                            "gating_state_dict": gating.state_dict(),
                            "n_sensors": n_sensors,
                            "hidden_dim": cfg.hidden_dim,
                        },
                        artifact_signature,
                    ),
                    save_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= cfg.iql_patience:
                    print(f"  Early stop @ epoch {epoch}")
                    break

        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        gating.load_state_dict(ckpt["gating_state_dict"])
    finally:
        predictor.train(predictor_was_training)
        for module, was_training, dropout in rnn_states:
            module.train(was_training)
            module.dropout = dropout
        for p in predictor.parameters():
            p.requires_grad_(True)

    print(f"[SoftAttention] done, best loss={best_loss:.4f}")
    return gating


if __name__ == "__main__":
    n, h, b = 179, 128, 4
    w = 60

    lr_model = LogisticRegressionPredictor(n, h)
    x = torch.randn(b, w, 2 * n)
    logit, tte = lr_model(x)
    print(f"LogisticPredictor logit: {logit.shape}, tte: {tte.shape}")

    gating = SoftAttentionGating(h, n)
    feat = torch.randn(b, h)
    weights = gating(feat)
    print(f"SoftAttention weights: {weights.shape}")
    mask = gating.select_top_m(feat, m=20)
    print(f"Top-M mask: {mask.shape}, sum={mask.sum(dim=1)}")
