import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """简单的CNN模型用于MNIST分类"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 经过两次池化后尺寸：28->14->7
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷积层1 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv1(x)))

        # 卷积层2 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv2(x)))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层1 + 激活函数 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 全连接层2（输出层）
        x = self.fc2(x)

        return x


class SimpleMLP(nn.Module):
    """简单的多层感知机（备选模型）"""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def create_model(model_name='cnn'):
    """创建模型工厂函数"""
    if model_name == 'cnn':
        return SimpleCNN()
    elif model_name == 'mlp':
        return SimpleMLP()
    else:
        raise ValueError(f"Unknown model name: {model_name}")