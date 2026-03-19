#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet18 训练脚本 - 野火检测
用于训练ResNet18模型进行二分类任务（wildfire/nowildfire）
"""

import os
import zipfile
import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI相关错误
import matplotlib.pyplot as plt
import numpy as np

# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WildfireDataset(Dataset):
    """野火检测数据集类"""
    def __init__(self, data_dir, transform=None, verify_images=False):
        """
        Args:
            data_dir: 包含'wildfire'和'nowildfire'子文件夹的路径
            transform: 可选的图像变换
            verify_images: 是否在初始化时验证图像（较慢但更安全）
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 类别映射
        class_map = {'wildfire': 1, 'nowildfire': 0}

        # 加载图像路径和标签
        for class_name, label in class_map.items():
            class_path = os.path.join(data_dir, class_name)

            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        # 如果启用验证，检查图像是否可读
                        if verify_images:
                            if self._verify_image(img_path):
                                self.image_paths.append(img_path)
                                self.labels.append(label)
                        else:
                            self.image_paths.append(img_path)
                            self.labels.append(label)
    
    def _verify_image(self, img_path):
        """验证图像文件是否可读"""
        try:
            img = Image.open(img_path)
            img.load()
            img.verify()  # 验证图像完整性
            return True
        except (IOError, OSError):
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 处理图像加载，包括截断的图像
        try:
            # 打开图像并转换为RGB
            image = Image.open(img_path)
            # 确保图像被完全加载
            image.load()
            image = image.convert('RGB')
        except (IOError, OSError) as e:
            # 如果图像损坏，尝试重新打开
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 尝试创建一个黑色图像作为占位符
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)

        return image, label


def extract_zips(zip_files, base_path='.'):
    """解压所有zip文件到对应的目录"""
    for zip_file in zip_files:
        zip_path = os.path.join(base_path, zip_file)
        if os.path.exists(zip_path):
            # 获取目录名（去掉.zip后缀）
            dir_name = zip_file.replace('.zip', '')
            dir_path = os.path.join(base_path, dir_name)
            # 创建目录（如果不存在）
            os.makedirs(dir_path, exist_ok=True)
            print(f"正在解压 {zip_file} 到 {dir_path}/...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dir_path)
            print(f"✓ {zip_file} 解压完成")
        else:
            print(f"⚠ 警告: {zip_file} 不存在")


def get_data_transforms():
    """获取数据预处理transform"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform


def create_model(num_classes=2, device='cpu'):
    """创建ResNet18模型"""
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # 修改最后一层以适应二分类任务
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    return model


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, val_recalls, epoch_times, output_dir):
    """绘制训练曲线并保存"""
    epochs = range(1, len(train_losses) + 1)
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ResNet18 Training Curves - Wildfire Detection', fontsize=16, fontweight='bold')
    
    # 1. Loss曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy曲线
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, marker='o')
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2, marker='s')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Recall曲线
    axes[1, 0].plot(epochs, val_recalls, 'g-', label='Val Recall', linewidth=2, marker='^')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Recall (%)', fontsize=12)
    axes[1, 0].set_title('Validation Recall (Wildfire Class)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Epoch Time曲线
    axes[1, 1].plot(epochs, epoch_times, 'm-', label='Epoch Time', linewidth=2, marker='d')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Epoch Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加平均时间信息
    avg_time = np.mean(epoch_times)
    axes[1, 1].axhline(y=avg_time, color='orange', linestyle='--', linewidth=1.5, 
                       label=f'Avg: {avg_time:.2f}s')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'\n✓ 训练曲线已保存到: {plot_path}')
    plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型，返回loss, accuracy和recall"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 用于计算混淆矩阵（二分类：0=nowildfire, 1=wildfire）
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测和标签用于计算recall
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * correct / total
    
    # 计算recall（对于wildfire类别，即label=1）
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # TP: 预测为wildfire(1)且实际为wildfire(1)
    # FN: 预测为nowildfire(0)但实际为wildfire(1)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    
    return epoch_loss, epoch_acc, recall * 100  # recall转换为百分比


def main(args):
    """主训练函数"""
    # 设置随机种子（可选，用于可复现性）
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 解压数据（如果需要）
    if args.extract:
        zip_files = ['train.zip', 'valid.zip', 'test.zip']
        extract_zips(zip_files, args.data_dir)
    
    # 检查数据目录
    base_path = args.data_dir
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'valid')
    test_dir = os.path.join(base_path, 'test')
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        print("错误: 数据目录不存在！请先解压数据或检查路径。")
        return
    
    # 获取数据transform
    train_transform, val_test_transform = get_data_transforms()
    
    # 创建数据集
    print("\n创建数据集...")
    # verify_images=False 可以加快加载速度，但可能遇到损坏的图像
    # 如果遇到很多图像错误，可以设置为 True
    train_dataset = WildfireDataset(train_dir, transform=train_transform, verify_images=args.verify_images)
    val_dataset = WildfireDataset(val_dir, transform=val_test_transform, verify_images=args.verify_images)
    test_dataset = WildfireDataset(test_dir, transform=val_test_transform, verify_images=args.verify_images)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    # 注意：在Windows或某些环境下，num_workers可能需要设置为0
    num_workers = args.num_workers if args.num_workers is not None else (0 if os.name == 'nt' else 2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(num_classes=2, device=device)
    print(f"模型结构已创建，最后一层输出维度 = 2")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_recalls = []
    epoch_times = []
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, val_recall = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_recalls.append(val_recall)
        
        # 更新学习率
        scheduler.step()
        
        # 计算epoch时间
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # 打印信息
        print(f'\nEpoch [{epoch+1}/{args.num_epochs}] ({epoch_time:.2f}s)')
        print(f'  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Recall: {val_recall:.2f}%')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, 'best_resnet18_model.pth')
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f'  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%) -> {model_path}')
        
        print("-" * 60)
    
    print(f'\n训练完成！最佳验证准确率: {best_val_acc:.2f}%')
    
    # 绘制并保存训练曲线
    print("\n生成训练曲线图...")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, val_recalls, epoch_times, args.output_dir)
    
    # 保存训练数据到CSV文件
    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc (%)', 'Val Loss', 'Val Acc (%)', 'Val Recall (%)', 'Epoch Time (s)'])
        for i in range(len(train_losses)):
            writer.writerow([
                i + 1,
                f'{train_losses[i]:.4f}',
                f'{train_accs[i]:.2f}',
                f'{val_losses[i]:.4f}',
                f'{val_accs[i]:.2f}',
                f'{val_recalls[i]:.2f}',
                f'{epoch_times[i]:.2f}'
            ])
    print(f'✓ 训练日志已保存到: {csv_path}')
    
    # 在测试集上评估
    if args.evaluate:
        print("\n" + "=" * 60)
        print("在测试集上评估模型...")
        print("=" * 60)
        
        model_path = os.path.join(args.output_dir, 'best_resnet18_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"已加载最佳模型: {model_path}")
        else:
            print("警告: 未找到保存的模型，使用当前模型进行评估")
        
        test_loss, test_acc, test_recall = validate(model, test_loader, criterion, device)
        print(f"\n测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  Recall: {test_recall:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet18 野火检测模型训练')
    
    # 数据相关参数
    parser.add_argument('--data-dir', type=str, default='.', 
                       help='数据目录路径（包含train/valid/test文件夹）')
    parser.add_argument('--extract', action='store_true',
                       help='是否解压zip文件（如果数据还未解压）')
    parser.add_argument('--verify-images', action='store_true',
                       help='在初始化时验证所有图像（较慢但更安全）')
    
    # 训练相关参数
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='训练轮数 (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批次大小 (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率 (default: 0.001)')
    parser.add_argument('--lr-step-size', type=int, default=7,
                       help='学习率衰减步长 (default: 7)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                       help='学习率衰减因子 (default: 0.1)')
    
    # 系统相关参数
    parser.add_argument('--num-workers', type=int, default=None,
                       help='数据加载器工作进程数 (None=自动选择)')
    parser.add_argument('--cpu', action='store_true',
                       help='强制使用CPU（即使有GPU可用）')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于可复现性）')
    
    # 输出相关参数
    parser.add_argument('--output-dir', type=str, default='.',
                       help='模型保存目录 (default: .)')
    parser.add_argument('--evaluate', action='store_true',
                       help='训练完成后在测试集上评估')
    
    args = parser.parse_args()
    
    main(args)
