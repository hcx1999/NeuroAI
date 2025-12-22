"""
LSNN (Liquid State Neural Network) 模型实现
基于论文: A solution to the learning dilemma for recurrent networks of spiking neurons

模型结构:
- 输入层: 将输入投影到循环层
- 循环层: 使用LIF神经元（可选SFA）的循环连接
- 输出层: 从循环层读取并输出分类结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .lif_neuron import LIFNeuron, LIFNeuronWithSFA


class LSNN(nn.Module):
    """
    LSNN (Liquid State Neural Network) 模型

    使用标准的BPTT进行训练
    """

    def __init__(
        self,
        input_size: int = 1,  # 输入维度（Sequential MNIST为1）
        hidden_size: int = 400,  # 循环层神经元数量
        output_size: int = 10,  # 输出类别数
        use_sfa: bool = False,  # 是否使用Spike-Frequency Adaptation
        tau_m: float = 20.0,  # 膜时间常数
        tau_adapt: float = 200.0,  # SFA时间常数（仅当use_sfa=True时使用）
        v_threshold: float = 1.0,  # 发放阈值
        v_reset: float = 0.0,  # 重置电位
        beta: float = 1.8,  # SFA强度（仅当use_sfa=True时使用）
        dt: float = 1.0,  # 时间步长
        dropout: float = 0.0,  # Dropout率
        readout_type: str = "rate",  # "rate" 或 "last": 如何从循环层读取输出
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_sfa = use_sfa
        self.readout_type = readout_type

        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size, bias=True)

        # 循环连接权重
        self.recurrent_weight = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.1
        )

        # 初始化循环权重（使用Xavier初始化）
        nn.init.xavier_uniform_(self.recurrent_weight)

        # LIF神经元
        if use_sfa:
            self.neuron = LIFNeuronWithSFA(
                tau_m=tau_m,
                tau_adapt=tau_adapt,
                v_threshold=v_threshold,
                v_reset=v_reset,
                beta=beta,
                dt=dt,
            )
        else:
            self.neuron = LIFNeuron(
                tau_m=tau_m,
                v_threshold=v_threshold,
                v_reset=v_reset,
                dt=dt,
            )

        # 输出层（readout）
        self.readout = nn.Linear(hidden_size, output_size, bias=True)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
        adapt_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_length, input_size]
            h_init: 初始隐藏状态（膜电位）[batch_size, hidden_size]
            adapt_init: 初始自适应变量（仅当use_sfa=True时使用）[batch_size, hidden_size]

        Returns:
            output: 输出logits [batch_size, output_size]
            hidden_states: 所有时间步的隐藏状态 [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length, _ = x.shape

        # 初始化状态
        if h_init is None:
            h = torch.zeros(
                batch_size, self.hidden_size, device=x.device, dtype=x.dtype
            )
        else:
            h = h_init

        if self.use_sfa:
            if adapt_init is None:
                adapt = torch.zeros(
                    batch_size, self.hidden_size, device=x.device, dtype=x.dtype
                )
            else:
                adapt = adapt_init

        # 存储所有时间步的隐藏状态（用于readout）
        hidden_states = []

        # 遍历时间步
        spikes_prev = torch.zeros(
            batch_size, self.hidden_size, device=x.device, dtype=x.dtype
        )

        for t in range(seq_length):
            # 输入投影
            x_t = x[:, t, :]  # [batch_size, input_size]
            input_current = self.input_projection(x_t)  # [batch_size, hidden_size]

            # 循环连接：使用前一个时间步的脉冲
            recurrent_current = torch.matmul(spikes_prev, self.recurrent_weight)

            # 总输入电流
            total_current = input_current + recurrent_current

            # 通过LIF神经元
            if self.use_sfa:
                spikes, h, adapt = self.neuron(total_current, h, adapt)
            else:
                spikes, h = self.neuron(total_current, h)

            # 更新前一个时间步的脉冲（用于下一个时间步的循环连接）
            spikes_prev = spikes

            # 存储隐藏状态（使用脉冲）
            hidden_states.append(spikes)

        # 堆叠所有时间步
        hidden_states = torch.stack(
            hidden_states, dim=1
        )  # [batch_size, seq_length, hidden_size]

        # Readout: 从循环层读取输出
        if self.readout_type == "rate":
            # 使用平均脉冲率
            hidden_mean = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        elif self.readout_type == "last":
            # 使用最后一个时间步
            hidden_mean = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"Unknown readout_type: {self.readout_type}")

        # 应用dropout
        hidden_mean = self.dropout(hidden_mean)

        # 输出层
        output = self.readout(hidden_mean)  # [batch_size, output_size]

        return output, hidden_states

    def reset_states(self):
        """重置神经元状态（用于新样本）"""
        # 这个方法主要用于非BPTT训练，BPTT会自动处理状态
        pass
