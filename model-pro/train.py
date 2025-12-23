import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
from tqdm import tqdm

from model import create_model
from utils import load_mnist_data, save_model, load_model


def train_epoch(model, device, train_loader, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    """测试模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 计算损失
            test_loss += criterion(output, target).item()

            # 统计准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    """主训练函数"""
    # 超参数设置
    config = {
        'model_name': 'cnn',  # 'cnn' 或 'mlp'
        'batch_size': 64,
        'epochs': 10,
        'learning_rate': 0.001,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_dir': './data',
        'save_dir': './checkpoints'
    }

    print("=" * 50)
    print("Training Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(config['seed'])

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 设备设置
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # 加载数据
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        download=True
    )

    # 创建模型
    print(f"Creating {config['model_name']} model...")
    model = create_model(config['model_name']).to(device)

    # 打印模型结构
    print(f"\nModel Architecture:")
    print(model)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练循环
    print("\nStarting training...")
    best_accuracy = 0

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, device, train_loader, criterion, optimizer, epoch
        )

        # 测试
        test_loss, test_acc = test(model, device, test_loader, criterion)

        # 学习率调整
        scheduler.step()

        # 计算时间
        epoch_time = time.time() - start_time

        # 打印结果
        print(f'\nEpoch {epoch}/{config["epochs"]}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.2f}s')

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            save_model(model, save_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")

    print("\n" + "=" * 50)
    print(f"Training completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print("=" * 50)

    # 保存最终模型
    final_save_path = os.path.join(config['save_dir'], 'final_model.pth')
    save_model(model, final_save_path)

    return model, best_accuracy


if __name__ == '__main__':
    model, best_acc = main()
