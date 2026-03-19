#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet18 测试脚本 - 野火检测
在 test/ 子目录上计算：
1) Accuracy
2) Recall（针对 wildfire 类别，即 label=1）
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile


# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WildfireDataset(Dataset):
    """野火检测数据集类（与训练脚本保持一致的数据结构：wildfire/nowildfire 两个子文件夹）"""

    def __init__(self, data_dir, transform=None, verify_images=False):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        class_map = {"wildfire": 1, "nowildfire": 0}

        for class_name, label in class_map.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue

            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(class_path, img_name)

                if verify_images:
                    if self._verify_image(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                else:
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def _verify_image(self, img_path):
        try:
            img = Image.open(img_path)
            img.load()
            img.verify()
            return True
        except (IOError, OSError):
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path)
            image.load()
            image = image.convert("RGB")
        except (IOError, OSError) as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model(device):
    # 不需要加载预训练权重：后续会 load 训练好的 state_dict
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model.to(device)


def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    # Recall(wildfire=1): TP / (TP + FN)
    tp = 0
    fn = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return accuracy, recall


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"使用设备: {device}")

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"找不到测试集目录: {test_dir}（需要包含 test/wildfire 和 test/nowildfire）")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")

    transform = get_val_transform()
    test_dataset = WildfireDataset(test_dir, transform=transform, verify_images=args.verify_images)

    num_workers = args.num_workers if args.num_workers is not None else (0 if os.name == "nt" else 2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"测试集大小: {len(test_dataset)}")

    model = create_model(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    test_acc, test_recall = evaluate(model, test_loader, device)

    print("\n测试集结果:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Recall (wildfire=1): {test_recall:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 野火检测测试脚本")

    parser.add_argument("--data-dir", type=str, default=".", help="数据目录路径（包含 test 文件夹）")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="训练得到的模型权重路径（如 best_resnet18_model.pth）",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小 (default: 64)")
    parser.add_argument("--num-workers", type=int, default=None, help="数据加载器工作进程数 (None=自动选择)")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU（即使有GPU可用）")
    parser.add_argument("--verify-images", action="store_true", help="初始化时验证所有测试图片（更慢但更安全）")

    args = parser.parse_args()
    main(args)

