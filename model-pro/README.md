# MNIST手写数字分类项目

使用PyTorch实现的简单MNIST分类器，包含CNN和MLP两种模型架构。

## 项目结构
mnist-classifier/<br>
├── model.py # 模型定义<br>
├── train.py # 训练脚本<br>
├── utils.py # 数据加载和工具函数<br>
├── data/ # MNIST数据集<br>
├── checkpoints/ # 模型保存目录<br>
├── requirements.txt # 依赖包<br>
└── README.md # 项目说明<br>

## 安装依赖
```python
pip install -r requirements.txt
```
1. 训练模型
```python
python train.py
```

