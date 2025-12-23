import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os


def load_mnist_data(data_dir='./data', batch_size=64, download=True):
    """
    加载MNIST数据集

    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        download: 是否下载数据

    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor，并归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])

    # 训练数据集
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    # 测试数据集
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def visualize_samples(data_loader, num_samples=10):
    """
    可视化数据集样本

    Args:
        data_loader: 数据加载器
        num_samples: 要显示的样本数量
    """
    # 获取一个批次的数据
    images, labels = next(iter(data_loader))

    # 创建子图
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i in range(num_samples):
        ax = axes[i]
        img = images[i].squeeze().numpy()  # 去掉通道维度
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # 打印数据统计信息
    print(f"Image shape: {images[0].shape}")
    print(f"Batch size: {len(images)}")
    print(f"Labels: {labels[:num_samples].numpy()}")


def save_model(model, path='model.pth'):
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")


def load_model(model, path='model.pth'):
    """加载模型"""
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model