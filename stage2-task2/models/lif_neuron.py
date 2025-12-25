"""
LIF (Leaky Integrate-and-Fire) 神经元实现
支持可选的 Spike-Frequency Adaptation (SFA)
基于论文: A solution to the learning dilemma for recurrent networks of spiking neurons
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire 神经元

    膜电位动态方程:
    tau_m * dv/dt = -v + I

    其中:
    - v: 膜电位
    - I: 输入电流
    - tau_m: 膜时间常数
    """

    def __init__(
        self,
        tau_m: float = 20.0,  # 膜时间常数 (ms)
        v_threshold: float = 1.0,  # 发放阈值
        v_reset: float = 0.0,  # 重置电位
        dt: float = 1.0,  # 时间步长 (ms)
        surrogate_function: Optional[nn.Module] = None,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.dt = dt
        self.detach_reset = detach_reset

        # 用于BPTT的surrogate gradient函数
        if surrogate_function is None:
            # 默认使用ATan作为surrogate gradient
            self.surrogate_function = self._atan_surrogate
        else:
            self.surrogate_function = surrogate_function

        # 衰减因子（注册为buffer，不参与训练但会移动到正确的设备）
        self.register_buffer("alpha", torch.tensor(math.exp(-dt / tau_m)))

    def _atan_surrogate(self, x: torch.Tensor) -> torch.Tensor:
        """ATan surrogate gradient function"""
        return 1.0 / (1.0 + (torch.pi * x).pow(2))

    def forward(
        self,
        x: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入电流 [batch_size, num_neurons]
            v: 初始膜电位 [batch_size, num_neurons]，如果为None则初始化为0

        Returns:
            spikes: 发放的脉冲 [batch_size, num_neurons]
            v_new: 新的膜电位 [batch_size, num_neurons]
        """
        if v is None:
            v = torch.zeros_like(x)

        # 更新膜电位: v_new = alpha * v + (1 - alpha) * I
        # 其中 alpha = exp(-dt/tau_m)
        v_new = self.alpha * v + (1 - self.alpha) * x

        # 计算是否发放脉冲
        spikes = (v_new >= self.v_threshold).float()

        # 重置已发放神经元的膜电位
        if self.detach_reset:
            v_new = v_new * (1 - spikes.detach()) + self.v_reset * spikes.detach()
        else:
            v_new = v_new * (1 - spikes) + self.v_reset * spikes

        return spikes, v_new


class LIFNeuronWithSFA(nn.Module):
    """
    带 Spike-Frequency Adaptation (SFA) 的 LIF 神经元

    增加了自适应阈值，使得神经元在频繁发放后阈值会暂时升高
    这增强了网络的计算能力，类似于LSTM的功能
    """

    def __init__(
        self,
        tau_m: float = 20.0,  # 膜时间常数
        tau_adapt: float = 200.0,  # 自适应时间常数
        v_threshold: float = 1.0,  # 基础发放阈值
        v_reset: float = 0.0,  # 重置电位
        beta: float = 1.8,  # SFA强度
        dt: float = 1.0,  # 时间步长
        surrogate_function: Optional[nn.Module] = None,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.tau_m = tau_m
        self.tau_adapt = tau_adapt
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.beta = beta
        self.dt = dt
        self.detach_reset = detach_reset

        if surrogate_function is None:
            self.surrogate_function = self._atan_surrogate
        else:
            self.surrogate_function = surrogate_function

        # 衰减因子（注册为buffer）
        self.register_buffer("alpha", torch.tensor(math.exp(-dt / tau_m)))
        self.register_buffer("alpha_adapt", torch.tensor(math.exp(-dt / tau_adapt)))

    def _atan_surrogate(self, x: torch.Tensor) -> torch.Tensor:
        """ATan surrogate gradient function"""
        return 1.0 / (1.0 + (torch.pi * x).pow(2))

    def forward(
        self,
        x: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        adapt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（带SFA）

        Args:
            x: 输入电流 [batch_size, num_neurons]
            v: 初始膜电位
            adapt: 初始自适应变量

        Returns:
            spikes: 发放的脉冲
            v_new: 新的膜电位
            adapt_new: 新的自适应变量
        """
        if v is None:
            v = torch.zeros_like(x)
        if adapt is None:
            adapt = torch.zeros_like(x)

        # 更新膜电位
        v_new = self.alpha * v + (1 - self.alpha) * x

        # 动态阈值 = 基础阈值 + beta * 自适应变量
        threshold = self.v_threshold + self.beta * adapt

        # 计算是否发放脉冲
        spikes = (v_new >= threshold).float()

        # 重置已发放神经元的膜电位
        if self.detach_reset:
            v_new = v_new * (1 - spikes.detach()) + self.v_reset * spikes.detach()
        else:
            v_new = v_new * (1 - spikes) + self.v_reset * spikes

        # 更新自适应变量: adapt_new = alpha_adapt * adapt + spikes
        adapt_new = self.alpha_adapt * adapt + spikes

        return spikes, v_new, adapt_new
