"""
Mask-Aware Risk Predictor (LSTM)

架构:
    InputProj(2N → H) → LSTM(H, num_layers) → LayerNorm
        ├─ 分类头 → logit   (训练用 BCEWithLogitsLoss)
        └─ 回归头 → tte_hat (Softplus 保证非负)

输入: (B, W, 2N)  — 拼接 [obs*mask, mask]
输出: logit (B,), tte_hat (B,)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


class MaskAwareRiskPredictor(nn.Module):
    """
    基于 LSTM 的 Mask-Aware 风险预测器。
    论文中使用 GRU 做信念编码；此处选用 LSTM，能力相当且更常用。
    """

    def __init__(
        self,
        n_sensors: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        lstm_dropout: float = 0.2,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        input_dim = 2 * n_sensors  # obs 拼接 mask

        # ---------- 输入投影 ----------
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ---------- LSTM 编码器 ----------
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        # ---------- 特征归一化 ----------
        self.feat_norm = nn.LayerNorm(hidden_dim)

        # ---------- 分类头: 输出未经 sigmoid 的 logit ----------
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ---------- 回归头: 输出非负 TTE ----------
        self.tte_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Xavier 初始化线性层，正交初始化 LSTM 权重。"""
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            elif param.dim() >= 2 and "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, W, 2N) — 输入序列 [obs*mask, mask]
        Returns:
            logit:   (B,)  未经 sigmoid 的分类 logit
            tte_hat: (B,)  非负的剩余时间预测
        """
        h = self.input_proj(x)                           # (B, W, H)
        lstm_out, _ = self.lstm(h)                       # (B, W, H)
        feat = lstm_out[:, -1, :]                        # (B, H) 取最后时间步
        feat = self.feat_norm(feat)                      # (B, H)

        logit = self.cls_head(feat).squeeze(-1)          # (B,)
        tte_hat = self.tte_head(feat).squeeze(-1)        # (B,)
        return logit, tte_hat

    # ------------------------------------------------------------------
    def encode_belief(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取信念状态向量（LSTM 最后时间步的隐藏表示）。
        用于 IQL 策略网络的输入。

        Args:
            x: (B, W, 2N) — 输入序列 [obs*mask, mask]
        Returns:
            feat: (B, hidden_dim) — 信念状态向量
        """
        h = self.input_proj(x)                           # (B, W, H)
        lstm_out, _ = self.lstm(h)                       # (B, W, H)
        feat = lstm_out[:, -1, :]                        # (B, H)
        feat = self.feat_norm(feat)                      # (B, H)
        return feat

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """推理接口，返回概率 p_hat 和 tte_hat。"""
        self.eval()
        logit, tte_hat = self.forward(x)
        p_hat = torch.sigmoid(logit)
        return p_hat, tte_hat


# ======================================================================
# 多任务损失函数
# ======================================================================

class MultiTaskLoss(nn.Module):
    """
    多任务损失 = cls_loss + lambda_tte * tte_loss

    - cls_loss : BCEWithLogitsLoss（数值稳定）
    - tte_loss : HuberLoss（对异常值鲁棒）
    """

    def __init__(self, lambda_tte: float = 0.5, huber_delta: float = 1.0,
                 pos_weight: float = None):
        super().__init__()
        self.lambda_tte = lambda_tte

        # BCEWithLogitsLoss 可选 pos_weight 处理类别不平衡
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(self, logit: torch.Tensor, tte_hat: torch.Tensor,
                label: torch.Tensor, tte_true: torch.Tensor):
        """
        Args:
            logit    : (B,) 分类 logit（未经 sigmoid）
            tte_hat  : (B,) 预测剩余时间
            label    : (B,) 真实标签 0/1
            tte_true : (B,) 真实剩余时间
        Returns:
            loss     : 标量
            cls_loss : 标量（用于日志）
            tte_loss : 标量（用于日志）
        """
        cls_loss = self.bce(logit, label)
        tte_loss = self.huber(tte_hat, tte_true)
        loss = cls_loss + self.lambda_tte * tte_loss
        return loss, cls_loss, tte_loss


# ======================================================================
# 快速 sanity check
# ======================================================================
if __name__ == "__main__":
    N, W, B = 36, 60, 4
    model = MaskAwareRiskPredictor(n_sensors=N)
    x = torch.randn(B, W, 2 * N)
    logit, tte = model(x)
    print(f"logit shape: {logit.shape}")   # (4,)
    print(f"tte   shape: {tte.shape}")     # (4,)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    p, t = model.predict(x)
    print(f"p_hat: {p}")
    print(f"tte_hat: {t}")

    # 测试损失函数
    criterion = MultiTaskLoss(lambda_tte=0.5)
    label = torch.tensor([0., 1., 0., 1.])
    tte_true = torch.tensor([100., 0., 50., 0.])
    loss, cls_l, tte_l = criterion(logit, tte, label, tte_true)
    print(f"loss={loss.item():.4f}  cls={cls_l.item():.4f}  tte={tte_l.item():.4f}")
    print("✓ Model & Loss sanity check passed.")
