import math
import sys
import time

import torch

import copy

import numpy as np
import matplotlib.pyplot as plt

import transforms
import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from .loss import KpLoss

####################################################调试显示

def generate_heatmaps(height, width, keypoints, sigma=2):
    num_kps = len(keypoints)
    heatmaps = np.zeros((num_kps, height, width), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0:
            continue
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        heatmap /= heatmap.max()
        heatmaps[i] = heatmap
    return heatmaps

def affine_points(points, mat):
    n = points.shape[0]
    homo_pts = np.hstack([points, np.ones((n, 1))])  # Nx3
    transformed = homo_pts @ mat.T  # Nx2
    return transformed

def visualize_local_heatmap(logits, targets, img_idx, k, scale=4, sigma=2, image_name=""):
    print(f"\n🔍 Visualizing heatmap local patch at epoch=_, step=_, image={image_name}")
    print(f"[🖼️] Image: {image_name}")

    with torch.no_grad():
        predict_kps0 = logits[img_idx].clone().detach().cpu().numpy()  # [num_kps, H, W]

        if "heatmap" in targets[img_idx]:
            gt_kps0 = targets[img_idx]["heatmap"].clone().detach().cpu().numpy()
        elif "keypoints" in targets[img_idx]:
            keypoints = targets[img_idx]["keypoints"]
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()

            H, W = predict_kps0.shape[1], predict_kps0.shape[2]
            input_h = targets[img_idx].get("image_height", H * scale)
            input_w = targets[img_idx].get("image_width", W * scale)

            print("Heatmap size:", H, W)
            print("Original image size:", input_w, input_h)
            print("Original Keypoints:", keypoints)

            # 如果有仿射变换矩阵，就先对关键点做仿射变换到网络输入尺寸（dst坐标系）
            if "trans" in targets[img_idx]:
                trans = targets[img_idx]["trans"]  # 2x3 仿射矩阵，正向变换
                keypoints = affine_points(keypoints[:, :2], trans)  # Nx2
                print("Affine transformed keypoints:", keypoints)
            else:
                print("Warning: no trans matrix found, keypoints assumed in input image coords")

            # 再缩放到heatmap尺寸
            keypoints_scaled = keypoints.copy()
            keypoints_scaled[:, 0] *= W / input_w
            keypoints_scaled[:, 1] *= H / input_h

            print("Scaled keypoints (for heatmap):", keypoints_scaled)

            gt_kps0 = generate_heatmaps(H, W, keypoints_scaled, sigma)
        else:
            raise ValueError("targets 中缺少 'heatmap' 或 'keypoints' 字段")

        pred_heatmap = predict_kps0[k]
        gt_heatmap = gt_kps0[k]

        center_y_pred, center_x_pred = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        center_y_gt, center_x_gt = np.unravel_index(np.argmax(gt_heatmap), gt_heatmap.shape)

        print(f"[Auto] Predicted max value location of k={k}: (x={center_x_pred}, y={center_y_pred})")
        print(f"[Auto] GroundTruth max value location of k={k}: (x={center_x_gt}, y={center_y_gt})")
        print(f"🔵 Predicted peak (heatmap): x={center_x_pred}, y={center_y_pred} → image: x={center_x_pred * scale}, y={center_y_pred * scale}")
        print(f"🟡 GroundTruth peak (heatmap): x={center_x_gt}, y={center_y_gt} → image: x={center_x_gt * scale}, y={center_y_gt * scale}")

        # === 仿射逆变换还原到原图坐标 ===
        if "reverse_trans" in targets[img_idx]:
            reverse_trans = targets[img_idx]["reverse_trans"]  # 2x3
            reverse_mat = np.vstack([reverse_trans, [0, 0, 1]])  # 3x3

            # pred_pt_input = np.array([center_x_pred * scale, center_y_pred * scale, 1.0])
            # gt_pt_input = np.array([center_x_gt * scale, center_y_gt * scale, 1.0])

            pred_pt_input = np.array([center_x_pred, center_y_pred, 1.0])
            gt_pt_input = np.array([center_x_gt, center_y_gt, 1.0])

            pred_pt_orig = reverse_mat @ pred_pt_input
            gt_pt_orig = reverse_mat @ gt_pt_input

            print(f"🔁 Predicted peak in original image: x={pred_pt_orig[0]:.2f}, y={pred_pt_orig[1]:.2f}")
            print(f"🎯 GroundTruth peak in original image: x={gt_pt_orig[0]:.2f}, y={gt_pt_orig[1]:.2f}")
        else:
            print("⚠️ warning: reverse_trans not found in targets → cannot recover original image coordinates.")

        # 截取局部窗口
        half_size = 10
        def crop_local(matrix, cx, cy):
            x_start = max(cx - half_size, 0)
            x_end = min(cx + half_size + 1, matrix.shape[1])
            y_start = max(cy - half_size, 0)
            y_end = min(cy + half_size + 1, matrix.shape[0])
            return matrix[y_start:y_end, x_start:x_end]

        pred_local = crop_local(pred_heatmap, center_x_pred, center_y_pred)
        gt_local = crop_local(gt_heatmap, center_x_gt, center_y_gt)

        def get_color(val):
            if val > 0.75:
                return 'red'
            elif val > 0.5:
                return 'orange'
            elif val > 0.25:
                return 'gray'
            else:
                return 'blue'

        def show_value_map(matrix, title="Heatmap Values"):
            H, W = matrix.shape
            fig, ax = plt.subplots(figsize=(W, H))
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            for i in range(H):
                for j in range(W):
                    val = matrix[i, j]
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            fontsize=10, color=get_color(val))
            ax.set_xticks(np.arange(W))
            ax.set_yticks(np.arange(H))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color='gray', linewidth=0.5)
            ax.set_title(title)
            plt.tight_layout()
            plt.show()

        print(f"Predicted local heatmap around predicted peak (size {pred_local.shape}):")
        print(pred_local)
        show_value_map(pred_local, title=f"Predicted Heatmap Local around Predicted Peak (k={k})")

        print(f"Ground truth local heatmap around GT peak (size {gt_local.shape}):")
        print(gt_local)
        show_value_map(gt_local, title=f"Ground Truth Heatmap Local around GT Peak (k={k})")



####################################################调试显示


####################################################训练监控
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            if torch.isnan(grad).any():
                print(f"⚠️ NaN detected in gradients of {name}")
                return False
            if torch.isinf(grad).any():
                print(f"⚠️ Inf detected in gradients of {name}")
                return False
            max_grad = grad.abs().max().item()
            if max_grad > 1e5:  # 阈值可调
                print(f"⚠️ Gradient too large in {name}: max_grad={max_grad}")
                return False
    return True

def check_weights(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️ NaN detected in weights of {name}")
            return False
        if torch.isinf(param).any():
            print(f"⚠️ Inf detected in weights of {name}")
            return False
    return True
####################################################训练监控

import math
import sys
import time
import torch
import transforms
import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from .loss import KpLoss

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=True, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    lr_scheduler = None
    if epoch == 0 and warmup:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mse = KpLoss()
    mloss = torch.zeros(1).to(device)
    for i, (images, targets, *rest) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.stack([img.to(device) for img in images])
        targets = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            losses = mse(outputs, targets)

        loss_dict_reduced = utils.reduce_dict({"losses": losses})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        mloss = (mloss * i + loss_value) / (i + 1)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # 同步所有进程日志
    metric_logger.synchronize_between_processes()
    if utils.is_main_process():
        print(f"Epoch summary: {metric_logger}")

    return mloss, optimizer.param_groups[0]["lr"]


@torch.no_grad()
def evaluate(model, data_loader, device, flip=False, flip_pairs=None):
    if flip:
        assert flip_pairs is not None, "flip enabled must provide flip_pairs"

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    key_metric = EvalCOCOMetric(data_loader.dataset.coco, "keypoints", "key_results.json", num_kpts=14)
    for images, targets, *rest in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack([img.to(device) for img in images])
        targets = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]

        if device.type != "cpu":
            torch.cuda.synchronize(device)

        outputs = model(images)

        if flip:
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, flip_pairs)
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., :-1]
            outputs = (outputs + flipped_outputs) * 0.5

        # decode keypoint
        # reverse_trans = [t["reverse_trans"] for t in targets]
        # outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

        reverse_trans = [t.get("reverse_trans", None) for t in targets]

        if reverse_trans[0] is None:
            # heatmap -> 原图坐标：* 4
            outputs = transforms.get_final_preds(outputs, trans=None, post_processing=True)
        else:
            outputs = transforms.get_final_preds(outputs, trans=reverse_trans, post_processing=True)

        key_metric.update(targets, outputs)
        metric_logger.update()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    key_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = key_metric.evaluate()
    else:
        coco_info = None

    return coco_info


