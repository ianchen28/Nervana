#!/usr/bin/env python3
"""
PyTorch 在 macOS M4 上的使用示例
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def create_simple_model():
    """创建一个简单的神经网络模型"""
    model = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 10),
                          nn.ReLU(), nn.Linear(10, 1))
    return model


def train_model():
    """训练一个简单的回归模型"""
    print("=== PyTorch 训练示例 ===")

    # 检查设备
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = create_simple_model().to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 创建训练数据 (y = x1 + x2 + noise)
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = (X[:, 0] + X[:, 1] + 0.1 * np.random.randn(1000)).reshape(-1, 1)

    # 转换为 PyTorch 张量
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练循环
    losses = []
    model.train()

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 测试模型
    model.eval()
    with torch.no_grad():
        test_X = torch.FloatTensor([[1, 1], [2, 3], [-1, 0]]).to(device)
        predictions = model(test_X)
        print(f"\n测试预测:")
        for i, (x, pred) in enumerate(
                zip(test_X.cpu().numpy(),
                    predictions.cpu().numpy())):
            print(f"输入: {x}, 预测: {pred[0]:.4f}, 实际: {x[0] + x[1]:.4f}")

    return losses


def plot_training_curve(losses):
    """绘制训练曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
    print("训练曲线已保存为 training_curve.png")


def main():
    """主函数"""
    print("PyTorch 在 macOS M4 上的机器学习示例")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MPS 支持: {torch.backends.mps.is_available()}")

    # 训练模型
    losses = train_model()

    # 绘制训练曲线
    plot_training_curve(losses)

    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
