# LSNN模型实现（使用BPTT训练）

基于论文：**A solution to the learning dilemma for recurrent networks of spiking neurons** (Nature Communications, 2020)

## 模型结构

本实现复刻了论文中的LSNN（Liquid State Neural Network）模型结构，但使用标准的BPTT（Backpropagation Through Time）算法进行训练，而不是论文中提出的e-prop算法。

### 模型组件

1. **LIF神经元** (`models/lif_neuron.py`)
   - 基础LIF神经元：实现泄漏积分-发放机制
   - 带SFA的LIF神经元：增加Spike-Frequency Adaptation，增强网络计算能力

2. **LSNN模型** (`models/lsnn.py`)
   - 输入层：将输入投影到循环层
   - 循环层：使用LIF神经元的循环连接网络
   - 输出层：从循环层读取并输出分类结果

## 文件结构

```
stage2-task2/
├── models/
│   ├── __init__.py          # 模块初始化
│   ├── lif_neuron.py        # LIF神经元实现
│   └── lsnn.py              # LSNN模型实现
├── data/
│   └── dataloader.py        # 数据加载器
├── train_lsnn_bptt.py       # 训练脚本（BPTT）
└── README.md                # 本文件
```

## 使用方法

### 基本训练

```bash
python train_lsnn_bptt.py
```

### 使用SFA（Spike-Frequency Adaptation）

```bash
python train_lsnn_bptt.py --use-sfa
```

### 自定义参数

```bash
python train_lsnn_bptt.py \
    --hidden-size 400 \
    --use-sfa \
    --tau-m 20.0 \
    --tau-adapt 200.0 \
    --beta 1.8 \
    --epochs 50 \
    --lr 1e-3 \
    --batch-size 32 \
    --clip-grad 1.0
```

## 主要参数说明

### 模型参数

- `--hidden-size`: 循环层神经元数量（默认：400）
- `--use-sfa`: 是否使用Spike-Frequency Adaptation
- `--tau-m`: 膜时间常数（默认：20.0 ms）
- `--tau-adapt`: SFA时间常数（默认：200.0 ms，仅当使用SFA时）
- `--v-threshold`: 发放阈值（默认：1.0）
- `--v-reset`: 重置电位（默认：0.0）
- `--beta`: SFA强度（默认：1.8，仅当使用SFA时）
- `--dt`: 时间步长（默认：1.0 ms）
- `--dropout`: Dropout率（默认：0.0）
- `--readout-type`: Readout类型，'rate'（平均脉冲率）或'last'（最后时间步）

### 训练参数

- `--epochs`: 训练轮数（默认：50）
- `--lr`: 学习率（默认：1e-3）
- `--weight-decay`: 权重衰减（默认：1e-4）
- `--clip-grad`: 梯度裁剪阈值（默认：1.0，0表示不裁剪）
- `--batch-size`: 批次大小（默认：32）
- `--lr-scheduler`: 使用学习率调度器
- `--lr-step`: 学习率衰减步长（默认：20）
- `--lr-gamma`: 学习率衰减因子（默认：0.5）

### 其他参数

- `--data-dir`: 数据集目录（默认：'data/sequential_mnist'）
- `--device`: 设备（'cuda'或'cpu'，默认：自动检测）
- `--save-dir`: 模型保存目录（默认：'checkpoints'）
- `--save-name`: 模型保存名称（默认：'lsnn_bptt'）

## 模型特点

1. **循环连接**：使用脉冲（spikes）进行循环连接，符合生物神经元的工作方式
2. **可选SFA**：支持Spike-Frequency Adaptation，增强网络计算能力
3. **BPTT训练**：使用标准的反向传播通过时间算法，支持梯度裁剪防止梯度爆炸
4. **灵活Readout**：支持使用平均脉冲率或最后时间步进行输出

## 注意事项

1. BPTT在处理长序列时内存消耗较大，如果遇到内存不足，可以：
   - 减小批次大小（`--batch-size`）
   - 使用梯度裁剪（`--clip-grad`）
   - 考虑使用Truncated BPTT（需要修改代码）

2. 使用SFA可以提升模型性能，但会增加计算开销

3. 模型会自动保存最佳模型和最终模型到`checkpoints`目录

## 与论文的区别

- **训练算法**：本实现使用BPTT，而论文提出的是e-prop算法
- **在线学习**：BPTT需要存储所有时间步的状态，是离线学习；e-prop是在线学习
- **生物合理性**：BPTT在生物学上不太合理，e-prop更符合生物学习机制

## 引用

如果使用本实现，请引用原论文：

```
Bellec, G., Scherr, F., Subramoney, A. et al. A solution to the learning dilemma 
for recurrent networks of spiking neurons. Nat Commun 11, 3625 (2020).
https://doi.org/10.1038/s41467-020-17236-y
```
