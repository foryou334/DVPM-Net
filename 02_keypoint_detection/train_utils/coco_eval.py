import json
import copy

from PIL import Image, ImageDraw
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .distributed_utils import all_gather, is_main_process
from transforms import affine_points


def merge(img_ids, eval_results):
    """将多个进程之间的数据汇总在一起"""
    all_img_ids = all_gather(img_ids)
    all_eval_results = all_gather(eval_results)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_results = []
    for p in all_eval_results:
        merged_eval_results.extend(p)

    merged_img_ids = np.array(merged_img_ids)

    # keep only unique (and in sorted order) images
    # 去除重复的图片索引，多GPU训练时为了保证每个进程的训练图片数量相同，可能将一张图片分配给多个进程
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_results = [merged_eval_results[i] for i in idx]

    return list(merged_img_ids), merged_eval_results


class EvalCOCOMetric:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = "keypoints",
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None,
                 threshold: float = 0.2,
                 num_kpts: int = 14):  # ✅ 加上 num_kpts 参数
        self.coco = copy.deepcopy(coco)
        self.obj_ids = []
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["keypoints"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name
        self.threshold = threshold
        self.num_kpts = num_kpts

        # ✅ 设置自定义 sigmas（SCU数据集 14 个点）
        self.custom_sigmas = np.array([
            0.5, 0.5, 0.5, 0.5,   # 左列
            0.8, 0.8, 0.8,        # 中左斜
            0.8, 0.8, 0.8,        # 中右斜
            0.5, 0.5, 0.5, 0.5    # 右列
        ]) / 10.0  # COCO标准要求除以 10

    def plot_img(self, img_path, keypoints, r=3):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for point in keypoints:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], fill=(255, 0, 0))
        img.show()

    def prepare_for_coco_keypoints(self, targets, outputs):
        for target, keypoints, scores in zip(targets, outputs[0], outputs[1]):
            if len(keypoints) == 0:
                continue
            obj_idx = int(target["obj_index"])
            if obj_idx in self.obj_ids:
                continue
            self.obj_ids.append(obj_idx)
            mask = np.greater(scores, 0.2)
            k_score = np.mean(scores[mask]) if mask.sum() else 0
            keypoints = np.concatenate([keypoints, scores], axis=1).reshape(-1)
            keypoints = [round(k, 2) for k in keypoints.tolist()]
            self.results.append({
                "image_id": target["image_id"],
                "category_id": 1,
                "keypoints": keypoints,
                "score": target["score"] * k_score
            })

    def update(self, targets, outputs):
        if self.iou_type == "keypoints":
            self.prepare_for_coco_keypoints(targets, outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def synchronize_results(self):
        eval_ids, eval_results = merge(self.obj_ids, self.results)
        self.aggregation_results = {"obj_ids": eval_ids, "results": eval_results}
        if is_main_process():
            with open(self.results_file_name, 'w') as f:
                f.write(json.dumps(eval_results, indent=4))

    def evaluate(self):
        if is_main_process():
            coco_true = self.coco
            coco_pre = coco_true.loadRes(self.results_file_name)
            self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)

            # ✅ 强制替换 COCOeval 默认 sigmas
            self.coco_evaluator.params.sigmas = self.custom_sigmas
            self.coco_evaluator.params.kpt_oks_sigmas = self.custom_sigmas
            self.coco_evaluator.params.useSegm = None  # 确保不使用分割

            self.coco_evaluator.evaluate()
            self.coco_evaluator.accumulate()
            print(f"IoU metric: {self.iou_type}")
            self.coco_evaluator.summarize()

            return self.coco_evaluator.stats.tolist()
        else:
            return None

