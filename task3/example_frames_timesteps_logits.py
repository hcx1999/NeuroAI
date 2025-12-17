"""
具体示例：帧序列、时间步和logits的关系

这个脚本展示了从event数据到最终logits的完整流程
"""

import torch
import torch.nn as nn

# ============================================
# 第一部分：Event数据 → 帧序列转换
# ============================================

print("=" * 60)
print("第一部分：Event数据转换为帧序列")
print("=" * 60)

# 假设我们有一个样本的原始event数据
# 原始event格式：(x, y, t, p) - 坐标、时间戳、极性
# 假设这个样本有 1000 个事件，分布在 128×128 的空间中

# 使用 frames_number=10, split_by="number" 进行切分
frames_number = 10
split_by = "number"
total_events = 1000

print(f"\n原始Event数据:")
print(f"  - 总事件数: {total_events}")
print(f"  - 空间分辨率: 128×128")
print(f"  - 事件格式: (x, y, t, p)")

print(f"\n帧切分参数:")
print(f"  - frames_number = {frames_number}")
print(f"  - split_by = '{split_by}'")

# 切分结果：每个帧包含 1000/10 = 100 个事件
events_per_frame = total_events // frames_number
print(f"  - 每帧事件数: {events_per_frame}")

# 转换为帧格式：[T, C, H, W]
# T = 10 (帧数)
# C = 2 (ON通道和OFF通道)
# H, W = 128, 128 (空间分辨率)

# 模拟帧数据（实际值会是0-1之间的浮点数，表示事件密度）
T, C, H, W = frames_number, 2, 128, 128
frame_sequence = torch.randn(T, C, H, W)  # 实际中会是事件累积的结果

print(f"\n转换后的帧序列形状:")
print(f"  - 形状: [{T}, {C}, {H}, {W}]")
print(f"  - 含义: [时间步数, 通道数(ON/OFF), 高度, 宽度]")
print(f"  - 每个帧: [{C}, {H}, {W}] = [{C}, 128, 128]")

# 展示单个帧的结构
print(f"\n单个帧 (frame[0]) 的结构:")
print(f"  - 形状: [{C}, {H}, {W}]")
print(f"  - 通道0 (ON事件): [{H}, {W}] 矩阵，值表示该位置ON事件的强度")
print(f"  - 通道1 (OFF事件): [{H}, {W}] 矩阵，值表示该位置OFF事件的强度")

# ============================================
# 第二部分：批处理后的数据形状
# ============================================

print("\n" + "=" * 60)
print("第二部分：DataLoader批处理")
print("=" * 60)

batch_size = 4  # 假设batch_size=4
# DataLoader会将多个样本组合成批次
batch_data = torch.randn(batch_size, T, C, H, W)

print(f"\n批处理后的数据形状:")
print(f"  - 输入形状: [{batch_size}, {T}, {C}, {H}, {W}]")
print(f"  - 含义: [批次大小, 时间步数, 通道数, 高度, 宽度]")
print(f"  - 具体: [4, 10, 2, 128, 128]")

print(f"\n批次中每个样本:")
for i in range(batch_size):
    print(f"  - 样本{i}: [{T}, {C}, {H}, {W}] = [10, 2, 128, 128]")

# ============================================
# 第三部分：模型Forward过程中的时间步处理
# ============================================

print("\n" + "=" * 60)
print("第三部分：模型Forward - 逐时间步处理")
print("=" * 60)

# 模拟模型处理过程
x = batch_data  # [4, 10, 2, 128, 128]
B, T, C, H, W = x.shape
print(f"\n输入到模型的数据:")
print(f"  - 形状: [{B}, {T}, {C}, {H}, {W}]")
print(f"  - 时间步数: {T}")

# 模拟简化的网络结构（用于演示）
# 实际网络: Conv2d(2, 64) -> ... -> Linear(512, 10)
print(f"\n模型结构（简化）:")
print(f"  - Conv1: 2通道 → 64通道")
print(f"  - Conv2: 64通道 → 128通道")
print(f"  - Conv3: 128通道 → 256通道")
print(f"  - FC1: 256*8*8 → 512")
print(f"  - FC2: 512 → 10 (10个类别)")

# 模拟每个时间步的处理
print(f"\n逐时间步处理过程:")
logits_accumulator = torch.zeros(B, 10)  # 累加器：[4, 10]

for t in range(T):
    print(f"\n--- 时间步 t={t} ---")
    
    # 提取当前时间步的帧
    x_t = x[:, t, :, :, :]  # [4, 10, 2, 128, 128] -> [4, 2, 128, 128]
    print(f"  提取帧 x_t: [{x_t.shape[0]}, {x_t.shape[1]}, {x_t.shape[2]}, {x_t.shape[3]}]")
    
    # 模拟通过网络（这里只是演示形状变化）
    # 实际中会经过: conv1 -> bn1 -> sn1 -> conv2 -> bn2 -> sn2 -> pool -> ...
    # 最终得到 logits: [B, 10]
    
    # 模拟logits（实际值会是网络计算的结果）
    logits_t = torch.randn(B, 10)  # [4, 10]
    print(f"  当前时间步的logits: [{logits_t.shape[0]}, {logits_t.shape[1]}]")
    print(f"  logits_t[0] (样本0的logits): {logits_t[0].tolist()}")
    
    # 累加到累加器
    logits_accumulator += logits_t
    print(f"  累加后的logits_acc[0]: {logits_accumulator[0].tolist()}")

# 计算平均logits（rate coding）
final_logits = logits_accumulator / float(T)
print(f"\n--- 最终结果 ---")
print(f"  平均logits (rate coding): [{final_logits.shape[0]}, {final_logits.shape[1]}]")
print(f"  最终logits[0] (样本0): {final_logits[0].tolist()}")

# ============================================
# 第四部分：具体数值示例
# ============================================

print("\n" + "=" * 60)
print("第四部分：具体数值示例（单个样本）")
print("=" * 60)

# 假设我们有一个样本，类别是"飞机"（类别0）
print(f"\n假设场景:")
print(f"  - 样本: 一个飞机的事件流")
print(f"  - 真实标签: 类别0 (airplane)")
print(f"  - frames_number: {frames_number}")

# 模拟10个时间步的logits
print(f"\n10个时间步的logits值（每个时间步都是[10]维向量）:")
time_step_logits = []
for t in range(T):
    # 模拟：每个时间步，模型对10个类别的"信心分数"
    logits = torch.randn(10) * 2  # 随机生成，实际是网络计算
    time_step_logits.append(logits)
    print(f"  t={t:2d}: {logits.tolist()}")

# 累加
accumulated = torch.zeros(10)
for t, logits in enumerate(time_step_logits):
    accumulated += logits
    print(f"  累加到t={t:2d}: {accumulated.tolist()}")

# 平均
final = accumulated / T
print(f"\n最终平均logits: {final.tolist()}")

# 预测类别
predicted_class = final.argmax().item()
print(f"预测类别: {predicted_class} (argmax of final logits)")

# ============================================
# 第五部分：可视化数据流
# ============================================

print("\n" + "=" * 60)
print("第五部分：完整数据流可视化")
print("=" * 60)

print(f"""
数据流转换过程:

1. 原始Event数据
   └─> 1000个事件: [(x₁,y₁,t₁,p₁), (x₂,y₂,t₂,p₂), ..., (x₁₀₀₀,y₁₀₀₀,t₁₀₀₀,p₁₀₀₀)]

2. 帧切分 (frames_number=10, split_by="number")
   ├─> Frame 0: 100个事件 → [2, 128, 128] 矩阵
   ├─> Frame 1: 100个事件 → [2, 128, 128] 矩阵
   ├─> Frame 2: 100个事件 → [2, 128, 128] 矩阵
   ├─> ...
   └─> Frame 9: 100个事件 → [2, 128, 128] 矩阵
   
   结果: [10, 2, 128, 128]

3. DataLoader批处理 (batch_size=4)
   └─> [4, 10, 2, 128, 128]

4. 模型Forward处理
   ├─> t=0: x[:,0,:,:,:] → [4, 2, 128, 128] → 网络 → [4, 10] logits₀
   ├─> t=1: x[:,1,:,:,:] → [4, 2, 128, 128] → 网络 → [4, 10] logits₁
   ├─> t=2: x[:,2,:,:,:] → [4, 2, 128, 128] → 网络 → [4, 10] logits₂
   ├─> ...
   └─> t=9: x[:,9,:,:,:] → [4, 2, 128, 128] → 网络 → [4, 10] logits₉
   
   累加: logits_acc = logits₀ + logits₁ + ... + logits₉
   平均: final_logits = logits_acc / 10
   
   结果: [4, 10] (4个样本，每个样本10个类别的分数)

5. 预测
   └─> argmax(final_logits, dim=1) → [4] (每个样本的预测类别)
""")

# ============================================
# 第六部分：关键概念对比
# ============================================

print("\n" + "=" * 60)
print("第六部分：关键概念对比")
print("=" * 60)

print(f"""
关键概念说明:

1. 帧序列 (Frame Sequence)
   - 定义: 将连续的事件流切分成离散的时间帧
   - 形状: [T, C, H, W] = [10, 2, 128, 128]
   - 含义: 10个时间帧，每帧是2通道的128×128图像
   - 来源: CIFAR10DVS数据集根据frames_number参数切分

2. 时间步 (Timestep)
   - 定义: 模型处理数据的时间维度
   - 数量: 等于帧序列的长度 (T = frames_number = 10)
   - 处理: 模型在每个时间步t处理对应的帧 x[:, t, :, :, :]
   - 作用: 让SNN能够处理时序信息

3. Logits
   - 定义: 模型对每个类别的原始分数（未经过softmax）
   - 形状: [B, num_classes] = [4, 10]
   - 每个时间步: 产生一个 [B, 10] 的logits
   - 累加: 将所有时间步的logits相加
   - 平均: 除以时间步数得到最终logits (rate coding)
   - 预测: argmax得到最终类别

关系图:
   帧序列 [B, T, C, H, W]
        ↓
   时间步循环 (t = 0 to T-1)
        ↓
   每步提取帧 [B, C, H, W]
        ↓
   通过网络
        ↓
   每步产生logits [B, 10]
        ↓
   累加所有时间步的logits
        ↓
   平均得到最终logits [B, 10]
        ↓
   argmax得到预测类别 [B]
""")

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)

