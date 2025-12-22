# 论文中的模型参数参考

基于论文 **"A solution to the learning dilemma for recurrent networks of spiking neurons"** (Nature Communications, 2020)

## 重要说明

论文中**没有明确列出所有具体的超参数值**，但根据论文描述和常见的LSNN实现，以下是参考参数设置：

## Sequential MNIST 任务参数

### 模型结构参数

根据论文描述和常见实现：

1. **循环层神经元数量 (hidden_size)**
   - 论文中提到了不同规模的网络
   - 常见设置：**400-800** 个神经元
   - 我们的默认值：**400**

2. **LIF神经元参数**
   - **膜时间常数 (tau_m)**: **20 ms** (论文中提到的典型值)
   - **发放阈值 (v_threshold)**: **1.0** (归一化后的值)
   - **重置电位 (v_reset)**: **0.0**
   - **时间步长 (dt)**: **1.0 ms**

3. **SFA参数** (当使用Spike-Frequency Adaptation时)
   - **适应时间常数 (tau_adapt)**: **200 ms** (论文中提到)
   - **适应强度 (beta)**: **1.8** (论文中提到的典型值)

### 训练超参数

论文中使用的是**e-prop算法**，而我们使用的是**BPTT**，因此训练参数可能有所不同：

1. **学习率 (learning rate)**
   - e-prop通常使用较小的学习率
   - BPTT建议：**1e-3 到 1e-4**
   - 我们的默认值：**1e-3**

2. **批次大小 (batch size)**
   - 论文中未明确说明
   - 常见设置：**32-128**
   - 我们的默认值：**32**

3. **训练轮数 (epochs)**
   - 论文中未明确说明
   - 通常需要：**50-100** 轮
   - 我们的默认值：**50**

4. **梯度裁剪 (gradient clipping)**
   - BPTT训练中很重要
   - 建议值：**1.0-5.0**
   - 我们的默认值：**1.0**

## 论文中提到的关键信息

1. **网络规模**：
   - 论文测试了不同规模的网络
   - 对于Sequential MNIST，通常使用中等规模（400-800神经元）

2. **SFA的重要性**：
   - 论文强调SFA显著提升了网络的计算能力
   - 使用SFA的网络性能接近LSTM

3. **权重初始化**：
   - 循环权重通常使用小的随机初始化
   - 我们的实现使用Xavier初始化

## 建议的参数配置

### 基础LSNN（无SFA）
```bash
python train_lsnn_bptt.py \
    --hidden-size 400 \
    --tau-m 20.0 \
    --v-threshold 1.0 \
    --v-reset 0.0 \
    --dt 1.0 \
    --lr 1e-3 \
    --batch-size 32 \
    --epochs 50 \
    --clip-grad 1.0
```

### LSNN with SFA（推荐）
```bash
python train_lsnn_bptt.py \
    --hidden-size 400 \
    --use-sfa \
    --tau-m 20.0 \
    --tau-adapt 200.0 \
    --beta 1.8 \
    --v-threshold 1.0 \
    --v-reset 0.0 \
    --dt 1.0 \
    --lr 1e-3 \
    --batch-size 32 \
    --epochs 50 \
    --clip-grad 1.0
```

### 大规模网络（如果内存允许）
```bash
python train_lsnn_bptt.py \
    --hidden-size 800 \
    --use-sfa \
    --tau-m 20.0 \
    --tau-adapt 200.0 \
    --beta 1.8 \
    --lr 5e-4 \
    --batch-size 16 \
    --epochs 100 \
    --clip-grad 1.0
```

## 参数调优建议

1. **如果训练不稳定**：
   - 减小学习率（如 5e-4）
   - 增加梯度裁剪阈值（如 2.0）
   - 减小批次大小

2. **如果性能不佳**：
   - 增加隐藏层大小（如 800）
   - 使用SFA（`--use-sfa`）
   - 增加训练轮数
   - 尝试不同的readout类型（`--readout-type last`）

3. **如果内存不足**：
   - 减小批次大小
   - 减小隐藏层大小
   - 使用梯度检查点（需要修改代码）

## 参考文献

论文原文中可能包含更详细的参数信息，建议：
1. 查看论文的**Methods部分**
2. 查看**Supplementary Information**
3. 查看论文的**GitHub仓库**（如果有）

论文链接：https://doi.org/10.1038/s41467-020-17236-y

