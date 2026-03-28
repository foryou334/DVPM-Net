import torch
import torch.nn.functional as F

class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.alpha = 0.1  #0.01~0.1

    def _get_pred_coords(self, heatmaps):
        B, K, H, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.view(B, K, -1)
        idx = torch.argmax(heatmaps_reshaped, dim=2)  # [B, K]
        pred_coords = torch.zeros((B, K, 2), device=heatmaps.device)
        pred_coords[:, :, 0] = (idx % W).float()      # x
        pred_coords[:, :, 1] = (idx // W).float()     # y
        return pred_coords

    def __call__(self, logits, targets, step=None):
        assert logits.dim() == 4, 'logits must be [B, K, H, W]'
        device = logits.device
        B, K, H, W = logits.shape

        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])  # [B, K, H, W]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])  # [B, K]
        kps_gt = torch.stack([torch.tensor(t["keypoints"], device=device) for t in targets])  # [B, K, 2]

        loss_hm = self.criterion(logits, heatmaps).mean(dim=[2, 3])  # [B, K]
        loss_hm = torch.sum(loss_hm * kps_weights) / B

        pred_coords = self._get_pred_coords(logits)  # [B, K, 2]

        # 宽高归一化坐标误差
        norm_pred_coords = pred_coords.clone()
        norm_pred_coords[:, :, 0] /= W
        norm_pred_coords[:, :, 1] /= H

        norm_kps_gt = kps_gt.clone()
        norm_kps_gt[:, :, 0] /= W
        norm_kps_gt[:, :, 1] /= H

        coord_loss = ((norm_pred_coords - norm_kps_gt) ** 2).mean(dim=2)  # [B, K]
        loss_coord = torch.sum(coord_loss * kps_weights) / B

        total_loss = loss_hm + self.alpha * loss_coord

        return total_loss
