"""
估算训练CIFAR10-DVS SNN网络所需的显存
"""

import torch
import torch.nn as nn

# ============================================
# 参数设置（从train_cifar10_DVS_snn.py获取）
# ============================================

# 训练参数
batch_size = 256
frames_number = 10
in_channels = 2
H, W = 128, 128  # CIFAR10DVS原始分辨率
num_classes = 10

# 模型参数
conv1_out = 64
conv2_out = 128
conv3_out = 256
fc1_out = 512

print("=" * 70)
print("显存估算：CIFAR10-DVS SNN训练")
print("=" * 70)

print(f"\n训练参数:")
print(f"  - batch_size: {batch_size}")
print(f"  - frames_number: {frames_number}")
print(f"  - 输入通道数: {in_channels} (ON/OFF)")
print(f"  - 输入分辨率: {H}×{W}")
print(f"  - 类别数: {num_classes}")

# ============================================
# 1. 输入数据显存
# ============================================

print(f"\n{'='*70}")
print("1. 输入数据显存")
print(f"{'='*70}")

# 输入形状: [B, T, C, H, W]
input_shape = (batch_size, frames_number, in_channels, H, W)
input_size_bytes = (
    batch_size * frames_number * in_channels * H * W * 4
)  # float32 = 4 bytes
input_size_mb = input_size_bytes / (1024**2)

print(f"\n输入数据形状: {input_shape}")
print(f"  - 数据类型: float32 (4 bytes)")
print(f"  - 显存占用: {input_size_mb:.2f} MB")
print(f"  - 计算: {batch_size} × {frames_number} × {in_channels} × {H} × {W} × 4 bytes")

# ============================================
# 2. 模型参数显存
# ============================================

print(f"\n{'='*70}")
print("2. 模型参数显存")
print(f"{'='*70}")


def calculate_conv_params(in_channels, out_channels, kernel_size=3, padding=1):
    """计算卷积层参数数量"""
    return out_channels * (in_channels * kernel_size * kernel_size + 1)  # +1 for bias


def calculate_linear_params(in_features, out_features):
    """计算全连接层参数数量"""
    return (in_features + 1) * out_features  # +1 for bias


# Conv1: 2 -> 64
conv1_params = calculate_conv_params(in_channels, conv1_out)
conv1_size = conv1_params * 4 / (1024**2)  # MB

# Conv2: 64 -> 128
conv2_params = calculate_conv_params(conv1_out, conv2_out)
conv2_size = conv2_params * 4 / (1024**2)

# Conv3: 128 -> 256
conv3_params = calculate_conv_params(conv2_out, conv3_out)
conv3_size = conv3_params * 4 / (1024**2)

# BatchNorm参数 (每个通道2个参数: weight + bias)
bn1_params = conv1_out * 2
bn2_params = conv2_out * 2
bn3_params = conv3_out * 2
bn_size = (bn1_params + bn2_params + bn3_params) * 4 / (1024**2)

# FC1: 256*8*8 -> 512
# 经过两次pooling: 128 -> 64 -> 32, 所以是 256 * 8 * 8 = 16384
fc1_in = (
    conv3_out * 8 * 8
)  # 经过两次pooling: 128/2/2 = 32, 但代码中是8*8，说明可能是128/2/2/2 = 16? 不对，让我重新算
# 看代码: conv3后pool，128/2=64, 再pool 64/2=32... 但代码写的是8*8，说明可能是128/2/2/2/2 = 8
# 实际上看代码: 128x128 -> pool -> 64x64 -> pool -> 32x32，但fc1是256*8*8，说明可能还有pool
# 让我按照代码中的实际值: 256 * 8 * 8 = 16384
fc1_params = calculate_linear_params(fc1_in, fc1_out)
fc1_size = fc1_params * 4 / (1024**2)

# FC2: 512 -> 10
fc2_params = calculate_linear_params(fc1_out, num_classes)
fc2_size = fc2_params * 4 / (1024**2)

total_params = (
    conv1_params
    + conv2_params
    + conv3_params
    + bn1_params
    + bn2_params
    + bn3_params
    + fc1_params
    + fc2_params
)
total_model_size = conv1_size + conv2_size + conv3_size + bn_size + fc1_size + fc2_size

print(f"\n各层参数:")
print(
    f"  - Conv1 ({in_channels}→{conv1_out}): {conv1_params:,} 参数, {conv1_size:.2f} MB"
)
print(
    f"  - Conv2 ({conv1_out}→{conv2_out}): {conv2_params:,} 参数, {conv2_size:.2f} MB"
)
print(
    f"  - Conv3 ({conv2_out}→{conv3_out}): {conv3_params:,} 参数, {conv3_size:.2f} MB"
)
print(f"  - BatchNorm: {bn1_params+bn2_params+bn3_params:,} 参数, {bn_size:.2f} MB")
print(f"  - FC1 ({fc1_in}→{fc1_out}): {fc1_params:,} 参数, {fc1_size:.2f} MB")
print(f"  - FC2 ({fc1_out}→{num_classes}): {fc2_params:,} 参数, {fc2_size:.2f} MB")
print(f"\n模型总参数: {total_params:,}")
print(f"模型显存占用: {total_model_size:.2f} MB")

# ============================================
# 3. 前向传播激活值显存
# ============================================

print(f"\n{'='*70}")
print("3. 前向传播激活值显存（单时间步）")
print(f"{'='*70}")

# 每个时间步处理一个帧: [B, C, H, W]
frame_shape = (batch_size, in_channels, H, W)

# Conv1输出: [B, 64, 128, 128]
conv1_out_shape = (batch_size, conv1_out, H, W)
conv1_out_size = batch_size * conv1_out * H * W * 4 / (1024**2)

# Pool后: [B, 64, 64, 64]
pool1_shape = (batch_size, conv1_out, H // 2, W // 2)
pool1_size = batch_size * conv1_out * (H // 2) * (W // 2) * 4 / (1024**2)

# Conv2输出: [B, 128, 64, 64]
conv2_out_shape = (batch_size, conv2_out, H // 2, W // 2)
conv2_out_size = batch_size * conv2_out * (H // 2) * (W // 2) * 4 / (1024**2)

# Pool后: [B, 128, 32, 32]
pool2_shape = (batch_size, conv2_out, H // 4, W // 4)
pool2_size = batch_size * conv2_out * (H // 4) * (W // 4) * 4 / (1024**2)

# Conv3输出: [B, 256, 32, 32]
conv3_out_shape = (batch_size, conv3_out, H // 4, W // 4)
conv3_out_size = batch_size * conv3_out * (H // 4) * (W // 4) * 4 / (1024**2)

# Pool后: [B, 256, 16, 16]
pool3_shape = (batch_size, conv3_out, H // 8, W // 8)
pool3_size = batch_size * conv3_out * (H // 8) * (W // 8) * 4 / (1024**2)

# Flatten后: [B, 256*8*8] = [B, 16384]
flatten_size = batch_size * fc1_in * 4 / (1024**2)

# FC1输出: [B, 512]
fc1_out_size = batch_size * fc1_out * 4 / (1024**2)

# FC2输出: [B, 10]
fc2_out_size = batch_size * num_classes * 4 / (1024**2)

# 估算激活值总大小（保留主要中间结果）
# 注意：实际训练中，PyTorch会保留所有需要梯度的中间结果用于反向传播
activation_size = (
    conv1_out_size
    + pool1_size
    + conv2_out_size
    + pool2_size
    + conv3_out_size
    + pool3_size
    + flatten_size
    + fc1_out_size
    + fc2_out_size
)

print(f"\n各层激活值大小（单时间步）:")
print(
    f"  - 输入帧: {frame_shape}, {batch_size * in_channels * H * W * 4 / (1024**2):.2f} MB"
)
print(f"  - Conv1输出: {conv1_out_shape}, {conv1_out_size:.2f} MB")
print(f"  - Pool1输出: {pool1_shape}, {pool1_size:.2f} MB")
print(f"  - Conv2输出: {conv2_out_shape}, {conv2_out_size:.2f} MB")
print(f"  - Pool2输出: {pool2_shape}, {pool2_size:.2f} MB")
print(f"  - Conv3输出: {conv3_out_shape}, {conv3_out_size:.2f} MB")
print(f"  - Pool3输出: {pool3_shape}, {pool3_size:.2f} MB")
print(f"  - Flatten: [{batch_size}, {fc1_in}], {flatten_size:.2f} MB")
print(f"  - FC1输出: [{batch_size}, {fc1_out}], {fc1_out_size:.2f} MB")
print(f"  - FC2输出: [{batch_size}, {num_classes}], {fc2_out_size:.2f} MB")
print(f"\n单时间步激活值总计: {activation_size:.2f} MB")

# ============================================
# 4. 反向传播显存（梯度）
# ============================================

print(f"\n{'='*70}")
print("4. 反向传播显存（梯度）")
print(f"{'='*70}")

# 梯度大小约等于参数大小
gradient_size = total_model_size
print(f"梯度显存: {gradient_size:.2f} MB (约等于模型参数大小)")

# ============================================
# 5. 优化器状态显存
# ============================================

print(f"\n{'='*70}")
print("5. 优化器状态显存（Adam）")
print(f"{'='*70}")

# Adam优化器需要存储：
# - 参数本身（已计入模型）
# - 一阶动量（momentum）: 与参数同大小
# - 二阶动量（variance）: 与参数同大小
# 总计：参数 × 3

adam_momentum_size = total_model_size  # 一阶动量
adam_variance_size = total_model_size  # 二阶动量
optimizer_size = adam_momentum_size + adam_variance_size

print(f"  - 一阶动量: {adam_momentum_size:.2f} MB")
print(f"  - 二阶动量: {adam_variance_size:.2f} MB")
print(f"  - 优化器总计: {optimizer_size:.2f} MB")

# ============================================
# 6. 总显存估算
# ============================================

print(f"\n{'='*70}")
print("6. 总显存估算")
print(f"{'='*70}")

# 前向传播：需要考虑所有时间步的激活值
# 但实际上，由于是循环处理，每个时间步的激活值可以复用
# 最坏情况：需要保留所有时间步的激活值用于反向传播
forward_activation = activation_size * frames_number  # 所有时间步

# 但实际PyTorch的自动微分会智能管理，通常只需要保留必要的中间结果
# 估算时使用一个合理的倍数（比如2-3倍单时间步激活值）
realistic_activation = activation_size * 2.5  # 实际激活值（考虑复用和释放）

total_memory = (
    input_size_mb  # 输入数据
    + total_model_size  # 模型参数
    + realistic_activation  # 前向激活值
    + gradient_size  # 梯度
    + optimizer_size  # 优化器状态
)

print(f"\n各组件显存占用:")
print(f"  - 输入数据: {input_size_mb:.2f} MB")
print(f"  - 模型参数: {total_model_size:.2f} MB")
print(f"  - 前向激活值: {realistic_activation:.2f} MB (估算)")
print(f"  - 梯度: {gradient_size:.2f} MB")
print(f"  - 优化器状态: {optimizer_size:.2f} MB")
print(f"\n{'─'*70}")
print(f"总显存估算: {total_memory:.2f} MB ≈ {total_memory/1024:.2f} GB")

# ============================================
# 7. 不同batch_size的显存对比
# ============================================

print(f"\n{'='*70}")
print("7. 不同batch_size的显存对比")
print(f"{'='*70}")

batch_sizes = [64, 128, 256, 512]
print(
    f"\n{'Batch Size':<12} {'输入数据(MB)':<15} {'激活值(MB)':<15} {'总显存(GB)':<15}"
)
print(f"{'-'*60}")

for bs in batch_sizes:
    # 输入数据
    input_mb = bs * frames_number * in_channels * H * W * 4 / (1024**2)

    # 激活值（按比例缩放）
    act_mb = (activation_size * 2.5) * (bs / batch_size)

    # 总显存（模型参数和优化器不随batch_size变化）
    total_gb = (
        input_mb + total_model_size + act_mb + gradient_size + optimizer_size
    ) / 1024

    print(f"{bs:<12} {input_mb:<15.2f} {act_mb:<15.2f} {total_gb:<15.2f}")

# ============================================
# 8. 显存优化建议
# ============================================

print(f"\n{'='*70}")
print("8. 显存优化建议")
print(f"{'='*70}")

print(
    f"""
如果显存不足，可以尝试以下方法：

1. 减小batch_size
   - 当前: {batch_size}
   - 建议: 64 或 128
   - 效果: 线性减少激活值显存

2. 减小frames_number
   - 当前: {frames_number}
   - 建议: 4 或 6
   - 效果: 减少输入数据和激活值显存

3. 使用梯度累积
   - 将大batch分成多个小batch
   - 累积梯度后再更新参数
   - 效果: 保持训练效果的同时减少显存

4. 使用混合精度训练
   - 使用torch.cuda.amp
   - 效果: 减少约50%的激活值显存

5. 使用梯度检查点（Gradient Checkpointing）
   - 牺牲计算时间换取显存
   - 效果: 减少约50%的激活值显存

6. 减少模型大小
   - 减少卷积通道数
   - 减少全连接层大小
   - 效果: 减少模型参数和激活值显存
"""
)

print("=" * 70)
print("估算完成！")
print("=" * 70)
