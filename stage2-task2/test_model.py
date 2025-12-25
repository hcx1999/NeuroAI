"""
测试LSNN模型是否能正常工作
"""

import torch
from models import LSNN


def test_lsnn():
    """测试基础LSNN模型"""
    print("=" * 60)
    print("测试基础LSNN模型（无SFA）")
    print("=" * 60)

    batch_size = 4
    seq_length = 784  # Sequential MNIST序列长度
    input_size = 1
    hidden_size = 100
    output_size = 10

    # 创建模型
    model = LSNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        use_sfa=False,
    )

    # 创建随机输入
    x = torch.randn(batch_size, seq_length, input_size)

    # 前向传播
    print(f"输入形状: {x.shape}")
    output, hidden_states = model(x)
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hidden_states.shape}")
    print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # 测试反向传播
    target = torch.randint(0, output_size, (batch_size,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"损失: {loss.item():.4f}")

    loss.backward()
    print("反向传播成功！")

    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"参数 {name} 有梯度，形状: {param.grad.shape}")
            break
    if has_grad:
        print("梯度计算成功！")
    else:
        print("警告：没有检测到梯度")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


def test_lsnn_sfa():
    """测试带SFA的LSNN模型"""
    print("\n" + "=" * 60)
    print("测试LSNN模型（带SFA）")
    print("=" * 60)

    batch_size = 4
    seq_length = 784
    input_size = 1
    hidden_size = 100
    output_size = 10

    # 创建模型
    model = LSNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        use_sfa=True,
        tau_m=20.0,
        tau_adapt=200.0,
        beta=1.8,
    )

    # 创建随机输入
    x = torch.randn(batch_size, seq_length, input_size)

    # 前向传播
    print(f"输入形状: {x.shape}")
    output, hidden_states = model(x)
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {hidden_states.shape}")
    print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # 测试反向传播
    target = torch.randint(0, output_size, (batch_size,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"损失: {loss.item():.4f}")

    loss.backward()
    print("反向传播成功！")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_lsnn()
    test_lsnn_sfa()
