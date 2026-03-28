import torch
import torch.nn.functional as F
from networkx import sigma
import numpy as np


class KPWeightedLoss(object):
    def __init__(self, sigma=2.0, alpha=0.5):
        """
        Args:
            sigma: 高斯核标准差（控制权重衰减范围）
            alpha: 真实点权重和预测点权重的混合比例 (0-1)
        """
        self.sigma = sigma   #2.0-5.0
        self.alpha = alpha   #0.5-0.8
        self.kernel_size = 2 * int(3 * sigma) + 1  # 覆盖99%能量的核尺寸

    def _create_gaussian_kernel(self, height, width, center, device):
        """为单个关键点生成空间权重图"""
        y = torch.arange(height, dtype=torch.float32, device=device)
        x = torch.arange(width, dtype=torch.float32, device=device)
        y, x = torch.meshgrid(y, x, indexing='ij')

        # 计算到中心点的距离
        dist = (x - center[0]) ** 2 + (y - center[1]) ** 2
        weights = torch.exp(-dist / (2 * self.sigma ** 2))
        return weights

    def _get_dynamic_weights(self, heatmaps, keypoints, device):
        """生成空间权重图"""
        batch_size, num_kps, height, width = heatmaps.shape
        dynamic_weights = torch.zeros_like(heatmaps)
        weighted_area = 0

        # 1. 基于真实关键点的权重
        for b in range(batch_size):
            for k in range(num_kps):
                center = (keypoints[b, k, 0], keypoints[b, k, 1])  # (x,y)
                dynamic_weights[b, k] = self._create_gaussian_kernel(
                    height, width, center, device)

        # 2. 基于异常高峰的权重
        with torch.no_grad():
            for b in range(batch_size):
                for k in range(num_kps):
                    highest_value = heatmaps[b, k].max()
                    error_peaks = (heatmaps[b, k] > highest_value * 0.75)  #所有大于0.75倍错误高值的峰索引
                    correct_peaks = (dynamic_weights[b, k] > 0.5)     #所有大于0.5倍真实高值的峰索引
                    false_positives = error_peaks & ~correct_peaks  #错误高值且不在真实高值内的峰索引
                    dynamic_weights[b, k][false_positives] = 0.5   #将错误高值权重设为0.5
                    weighted_area += len(false_positives.nonzero())  #记录错误高值数量

        # 3. 基于预测关键点的权重（动态调整）
        with torch.no_grad():
            pred_kps = self._get_predicted_keypoints(heatmaps)
            for b in range(batch_size):
                for k in range(num_kps):
                    center = (pred_kps[b, k, 0], pred_kps[b, k, 1])
                    pred_weights = self._create_gaussian_kernel(
                        height, width, center, device)
                    # 混合真实点和预测点的权重
                    dynamic_weights[b, k] = self.alpha * dynamic_weights[b, k] + \
                                            (1 - self.alpha) * pred_weights

        weighted_area = weighted_area + (9 * np.pi * self.sigma**2)* 2  #高斯核包含99.7%能量的区域半径为3σ，因此面积为9πσ²

        return dynamic_weights,weighted_area

    def _get_predicted_keypoints(self, heatmaps):
        """从热图中提取预测关键点坐标"""
        batch_size, num_kps, height, width = heatmaps.shape
        heatmaps_flat = heatmaps.view(batch_size, num_kps, -1)
        indices = heatmaps_flat.argmax(dim=-1)
        pred_kps = torch.zeros(batch_size, num_kps, 2, device=heatmaps.device)
        pred_kps[:, :, 0] = indices % width  # x坐标
        pred_kps[:, :, 1] = indices // width  # y坐标
        return pred_kps

    def __call__(self, logits, targets):
        """
        Args:
            logits: 网络输出 [B, num_kps, H, W]
            targets: 包含heatmap和keypoints的列表
        Returns:
            加权后的损失值
        """
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        batch_size = logits.shape[0]

        # 获取目标热图和关键点
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        keypoints = torch.stack([torch.tensor(t["keypoints"]).to(device) for t in targets])  # [B, num_kps, 3]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])  # [B, num_kps]

        # 生成空间权重图 [B, num_kps, H, W]
        dynamic_weights , weighted_area= self._get_dynamic_weights(heatmaps, keypoints, device)

        # 计算加权损失
        loss_per_pixel = F.mse_loss(logits, heatmaps, reduction='none')
        weighted_loss = loss_per_pixel * dynamic_weights * 20

        # 按关键点权重加权
        loss_per_kp = (loss_per_pixel.mean(dim=[2, 3]) + weighted_loss.sum(dim=(-2,-1))/weighted_area )  # [B, num_kps]
        total_loss = loss_per_kp.sum() / (batch_size + 1e-6)


        print('loss_per_pixel:', (loss_per_pixel.mean(dim=[2,3])).sum()/(batch_size + 1e-6))
        print('weighted_loss:', (weighted_loss.sum(dim=[-2,-1])/weighted_area).sum()/(batch_size + 1e-6) )
        # # logits为输出热图,断点不显示着色单元，故复制到变量predict_kps0来观察，targets['heatmap']为真实热图
        # print(logits[0][0][37:42, 39:44])
        # print('kps:', logits[0][0][39][41].item())
        # predict_kps0 = logits[0]
        # predict_kps0 = predict_kps0.cpu().detach().numpy() * 10
        # # 获取整个二维张量的最大值及其索引
        # max_value, max_index_flat = torch.max(logits[0][0].view(-1), dim=0)
        # # 将一维索引转换为二维坐标
        # max_index_row = max_index_flat // logits[0][0].size(1)
        # max_index_col = max_index_flat % logits[0][0].size(1)
        # print("最大值:", max_value.item())
        # print("最大值的索引（二维）: 行 {}, 列 {}".format(max_index_row.item(), max_index_col.item()))


        return total_loss