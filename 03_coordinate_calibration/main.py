from tkinter import image_names

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import warnings
import matplotlib.pyplot as plt
import matplotlib
import os
import re

# 防止中文乱码
matplotlib.rc("font", family='Microsoft YaHei')
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# ========== Config ==========
INPUT_SOURCE = './data/data_1.xlsx'

OUTPUT_EXCEL = './runs/only_predect/data_1/only_predictions.xlsx'

MODEL_PATH = "./weight/data_1/best_model.pth"

BATCH_SIZE = 256
LOSS_TYPE = "l1"   # 和训练时保持一致
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========== Data Processing ==========
class AngleDataProcessor:
    def load_data(self):
        # 读取包含预测值和真实值的 Excel 文件
        pred_df = pd.read_excel(INPUT_SOURCE)

        image_names = pred_df["image_name"].values

        # 读取预测值列 (sin, cos) - 预测值位于 J-K-L-M-N-O 列
        X_raw = pred_df[["J", "K", "L", "M", "N", "O"]].values.astype(np.float32)  # 预测值

        # *系数，降低敏感性#####################################################################################
        X_all = X_raw / 1

        return image_names, X_all



# ========== Model ==========
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.norm2(x)
        return x + residual


class GatedResidualBlock(nn.Module):
    """轻量 gating 的残差块"""
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = ResidualBlock(dim, dropout_rate)
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate_val = self.gate(x)  # [batch, 1]
        return x + gate_val * self.block(x)


class MultiHeadOutput(nn.Module):
    """多头输出层：多个 head + softmax 权重融合"""
    def __init__(self, hidden_dim, output_dim, num_heads=3):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_heads)])
        self.gate = nn.Linear(hidden_dim, num_heads)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=-1)  # [batch, heads]
        outputs = torch.stack([head(x) for head in self.heads], dim=1)  # [batch, heads, output_dim]
        return torch.sum(weights.unsqueeze(-1) * outputs, dim=1)  # 加权融合


class SelfAttentionBlock(nn.Module):
    """简单的多头自注意力模块 + 残差"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: [batch, dim]
        x = x.unsqueeze(1)  # -> [batch, seq=1, dim]
        attn_out, _ = self.attn(x, x, x)  # 自注意力
        x = x + attn_out
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)
        return x.squeeze(1)  # -> [batch, dim]


# ========== 1D Convolution Block ==========
class ConvBlock(nn.Module):
    """轻量级 1D 卷积 + 残差 + 激活"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # 将 [batch_size, dim] 转为 [batch_size, dim, 1] 以便卷积
        x = x.unsqueeze(2)  # 变为 [batch_size, dim, 1]
        x = self.conv(x)  # 卷积操作
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        # 将 [batch_size, dim, 1] 恢复为 [batch_size, dim]
        x = x.squeeze(2)  # 变回 [batch_size, dim]
        return x + residual  # 加入残差连接


# ========== 改进后的模型（加入1D卷积） ==========
class ImprovedMLP_Conv_Attn(nn.Module):
    """MLP + 1D Convolution + Self-Attention 模块"""

    def __init__(self, input_dim=6, hidden_dim=1024, output_dim=6,
                 num_blocks=3, dropout_rate=0.2, num_heads=3, attn_heads=4, conv_channels=512):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # 卷积块部分
        self.conv_block = ConvBlock(hidden_dim, conv_channels, kernel_size=3, dropout=dropout_rate)

        # 残差块
        self.blocks = nn.Sequential(
            *[GatedResidualBlock(conv_channels, dropout_rate) for _ in range(num_blocks)]
        )

        # 自注意力机制
        self.attn_block = SelfAttentionBlock(conv_channels, num_heads=attn_heads, dropout=dropout_rate)

        # 输出部分
        self.output_fc = MultiHeadOutput(conv_channels, output_dim, num_heads=num_heads)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.input_fc(x))

        # 通过卷积模块提取局部特征
        x = self.conv_block(x)

        # 然后通过残差块处理
        x = self.blocks(x)

        # 再通过自注意力机制处理全局特征
        x = self.attn_block(x)

        # 最终通过输出层
        x = self.output_fc(x)
        return x


# ========== Loss Functions ==========
class L1L2Loss(nn.Module):
    """混合 L1 + MSE"""
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + (1 - self.alpha) * self.l2(pred, target)


class ExpL1Loss(nn.Module):
    """
    指数型误差放大：mean(expm1(beta * |e|))
    对小误差近似 beta*|e|，对大误差迅速变大
    """
    def __init__(self, beta=50.0, max_arg=20.0):
        super().__init__()
        self.beta = beta
        self.max_arg = max_arg  # 防止溢出：beta*|e| 会被 clamp 到 [0, max_arg]

    def forward(self, pred, target):
        abs_e = torch.abs(pred - target)
        arg = torch.clamp(self.beta * abs_e, max=self.max_arg)
        return torch.mean(torch.expm1(arg))  # exp(arg)-1，零点处梯度更平滑


def get_loss_fn(loss_type="l1", **kwargs):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "smoothl1":
        return nn.SmoothL1Loss(beta=0.5)
    elif loss_type == "mix":
        return L1L2Loss(alpha=0.7)
    elif loss_type == "exp":    # 新增指数型损失
        beta = kwargs.get("beta", 7.0)
        max_arg = kwargs.get("max_arg", 7.0)
        return ExpL1Loss(beta=beta, max_arg=max_arg)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ========== Evaluation ==========
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            # batch 可能是单元素 tuple
            xb = batch[0].to(device)
            pred = model(xb)
            predictions.append(pred.cpu().numpy())

    predictions = np.vstack(predictions)
    return predictions



# ========== Main ==========
def main():
    processor = AngleDataProcessor()

    # 读取数据，同时标记哪些是 [404,404,404]
    image_names, X_all = processor.load_data()


    # 创建 DataLoader
    te_ds = TensorDataset(torch.FloatTensor(X_all))
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE)

    # 初始化模型（和训练时保持一致）
    model = ImprovedMLP_Conv_Attn(
        input_dim=X_all.shape[1],
        output_dim=6,  # 输出 6 维 sin/cos
        hidden_dim=1024,
        num_blocks=3,
        num_heads=3,
        attn_heads=4,
        conv_channels=1024,
        dropout_rate=0.2
    ).to(device)

    # 加载训练好的权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 执行预测
    all_preds = []
    with torch.no_grad():
        for batch in te_loader:
            xb = batch[0].to(device)
            pred = model(xb)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.vstack(all_preds)

    # /系数#############################################################################################
    preds = all_preds * 1

    # 检查输出目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(OUTPUT_EXCEL)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    def sort_image_names(image_names):

        return sorted(image_names, key=lambda x: int(re.search(r'(\d+)', str(x)).group()) if isinstance(x, str) else 0)


    # 排序后的 image_names
    sorted_image_names = sort_image_names(image_names)

    # 保存到 Excel
    result_df = pd.DataFrame(
        np.hstack([np.array(sorted_image_names).reshape(-1, 1), preds]),
        columns=["image_name", "Pred_yaw_sin","Pred_yaw_cos", "Pred_pitch_sin","Pred_pitch_cos","Pred_roll_sin","Pred_roll_cos"]
    )

    result_df["Pred_yaw_sin"] = pd.to_numeric(result_df["Pred_yaw_sin"], errors='coerce')
    result_df["Pred_yaw_cos"] = pd.to_numeric(result_df["Pred_yaw_cos"], errors='coerce')
    result_df["Pred_pitch_sin"] = pd.to_numeric(result_df["Pred_pitch_sin"], errors='coerce')
    result_df["Pred_pitch_cos"] = pd.to_numeric(result_df["Pred_pitch_cos"], errors='coerce')
    result_df["Pred_roll_sin"] = pd.to_numeric(result_df["Pred_roll_sin"], errors='coerce')
    result_df["Pred_roll_cos"] = pd.to_numeric(result_df["Pred_roll_cos"], errors='coerce')

    result_df["Pred_yaw"] = np.arctan2(result_df["Pred_yaw_sin"], result_df["Pred_yaw_cos"]) * 180 / np.pi
    result_df["Pred_pitch"] = np.arctan2(result_df["Pred_pitch_sin"], result_df["Pred_pitch_cos"]) * 180 / np.pi
    result_df["Pred_roll"] = np.arctan2(result_df["Pred_roll_sin"], result_df["Pred_roll_cos"]) * 180 / np.pi


    # 保存结果到 Excel 文件
    result_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Predictions saved to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
