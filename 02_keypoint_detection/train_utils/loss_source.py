import torch
import matplotlib.pyplot as plt
import numpy as np


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs

        # # ================== 可视化调试 ====================
        # with torch.no_grad():
        #     print('logits热图局部窗口：')
        #     print(logits[0][0][50:60, 523:533])  # 可根据实际关键点位置修改坐标
        #     print('logits某点值:', logits[0][0][39][41].item())
        #
        #     predict_kps0 = logits[0].clone().detach().cpu().numpy()  # [num_kps, H, W]
        #     gt_kps0 = targets[0]["heatmap"].clone().detach().cpu().numpy()  # [num_kps, H, W]
        #
        #     k = 0  # 显示第 k 个关键点
        #     pred_heatmap = predict_kps0[k]
        #     gt_heatmap = gt_kps0[k]
        #
        #     H, W = pred_heatmap.shape
        #
        #     def get_color(val):
        #         if val > 0.75:
        #             return 'red'
        #         elif val > 0.5:
        #             return 'orange'
        #         elif val > 0.25:
        #             return 'gray'
        #         else:
        #             return 'blue'
        #
        #     def show_value_map(matrix, title="Heatmap Values"):
        #         fig, ax = plt.subplots(figsize=(W / 2, H / 2))
        #         ax.set_xlim(-0.5, W - 0.5)
        #         ax.set_ylim(H - 0.5, -0.5)
        #
        #         for i in range(H):
        #             for j in range(W):
        #                 val = matrix[i, j]
        #                 ax.text(j, i, f"{val:.2f}", ha='center', va='center',
        #                         fontsize=6, color=get_color(val))
        #
        #         ax.set_xticks(np.arange(W))
        #         ax.set_yticks(np.arange(H))
        #         ax.set_xticklabels([])
        #         ax.set_yticklabels([])
        #         ax.grid(True, color='gray', linewidth=0.5)
        #         ax.set_title(title)
        #         plt.tight_layout()
        #         plt.show()
        #
        #     # 🔹 显示预测热图值
        #     show_value_map(pred_heatmap, title=f"Predicted Heatmap (k={k})")
        #
        #     # 🔹 显示真实热图值
        #     show_value_map(gt_heatmap, title=f"Ground Truth Heatmap (k={k})")


        return loss
