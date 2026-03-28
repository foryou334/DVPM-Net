DVPM-Net: Deep Vision Pose Measurement Network for Robotic Drilling
Official implementation of the paper: "Deep Learning-Driven Pose Measurement Framework for Robotic Drilling End-Effectors in Aerospace Manufacturing".

📌 Overview
This project provides a robust deep learning-driven framework designed for high-precision pose measurement in automated aircraft assembly. Our system addresses the challenges of variable illumination and non-linear error accumulation in large-scale industrial environments (4–7 meters).

Key Features:
DVPM-Net: A multi-task architecture for illumination-invariant semantic keypoint extraction.

🏗️ System Architecture
The framework consists of three main stages:

Target Detection： Detect the ROI of the cooperative target using YOLOv5, and perform cropping and padding to a unified size.

Feature Extraction: Using DVPM-Net to localize high-resolution semantic keypoints on the drilling end-effector.

Spatial Mapping: Implementing CTC-Net to mitigate error propagation during coordinate transformation.


📂 Project Structure
DVPM-Net/
├── 01_object_identification/      # YOLOv5-based target ROI detection
├── 02_keypoint_detection/         # HRNet-based keypoint localization (DVPM-Net)
│   └── weights/                   # Model weights (Download link in url.txt)
├── 03_coordinate_calibration/      # CTC-Net for robot-camera mapping
├── data/                          # Sample dataset (Laboratory version)
└── README.md

📊 Dataset Description
Important Notice on Data Privacy:
Due to strict confidentiality agreements regarding industrial aerospace manufacturing, the full industrial-scale dataset used in our paper cannot be made public.

However, we provide a Laboratory Dataset collected in a controlled environment. This dataset contains similar feature geometries and can be used to verify the algorithm's logic and performance.

Laboratory Data: Available in the data/SCU.

Industrial Data: Restricted (Confidential).


🚀 Getting Started

Weights
The weights for hr_asff_coor_loss0.1.pth and yolo_best.pt exceed GitHub's size limit.
Please download them from the link provided in:
02_keypoint_detection/weights/url.txt

📝 Citation
If you find this work helpful for your research, please cite our paper:
High-Precision Monocular Pose Measurement of Drilling Robot End-Effectors via Deep Learning
