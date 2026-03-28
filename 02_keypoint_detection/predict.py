import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms


def predict_all_person():
    # TODO
    pass


def predict_single_person():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = True
    resize_hw = (960,1280)
    # img_path = r"E:\PythonProjects\Wz-HRNet\data\SCU\test\200.jpg"
    img_path = r"E:\PythonProjects\Wz-HRNet\data\img\img_7m\2\_22.jpg"
    weights_path = "./save_weights/model-209.pth"
    keypoint_json_path = "target_keypoints.json"

    padding_csv_path = ".csv/padding/test_padding.csv"  # CSV_padding

    offset_csv_path = ".csv/yolo/test_yolo.csv"  # CSV_yolo
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."
    assert os.path.exists(padding_csv_path), f"file: {padding_csv_path} does not exist."
    assert os.path.exists(offset_csv_path), f"file: {offset_csv_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        # transforms.AffineTransform(scale=(0.8, 1.2), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        outputs = model(img_tensor.to(device))

        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        print("keypoints:\n", keypoints)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.7, r=16)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("./single_test/test_result.jpg")

# === 从CSV中读取该图像偏移并恢复关键点位置 ===
        # 读取 Tab 分隔的 CSV
        padding_csv = pd.read_csv(padding_csv_path)
        offset_csv = pd.read_csv(offset_csv_path)

        # 清洗图像名（去掉空格、回车换行、小写）
        padding_csv["image_name"] = padding_csv["image_name"].astype(str).str.strip().str.lower()
        offset_csv["image_name"] = offset_csv["image_name"].astype(str).str.strip().str.replace(r"[\r\n]+", "",regex=True).str.lower()

        # 当前图像名
        img_name = os.path.basename(img_path).strip().lower()

        # 查找padding对应的偏移值
        padding_row = padding_csv[padding_csv["image_name"] == img_name]
        assert not padding_row.empty, f"No padding offset found for {img_name} in {padding_csv_path}"
        pad_x = float(padding_row["offset_x"].values[0])
        pad_y = float(padding_row["offset_y"].values[0])

        # 查找yolo对应的偏移值
        offset_row = offset_csv[offset_csv["image_name"] == img_name]
        assert not offset_row.empty, f"No offset found for {img_name} in {offset_csv_path}"

        x_offset = float(offset_row["x"].values[0])
        y_offset = float(offset_row["y"].values[0])

        # ==== 关键点三步恢复 ====
        keypoints[:, 0] -= pad_x
        keypoints[:, 1] -= pad_y

        # 反向补偿 gain 和 pad，恢复到原始bbox区域内的坐标
        # #yolo裁剪时默认gain=1.02：原框放大 2%；pad=10：在放大后的框周围再加10像素padding；
        gain = 1.02
        pad = 10.0

        # 从裁图内坐标 -> bbox原始框坐标
        keypoints[:, 0] = (keypoints[:, 0] - pad) / gain
        keypoints[:, 1] = (keypoints[:, 1] - pad) / gain

        # 再加上bbox左上角偏移，恢复为原图坐标
        keypoints[:, 0] += x_offset
        keypoints[:, 1] += y_offset

        print("Restored keypoints (in original image):\n", keypoints)
        print("scores:\n", scores)


if __name__ == '__main__':
    predict_single_person()
