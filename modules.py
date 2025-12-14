import torch
import torch.nn as nn
from surrogate import SuperSpikeFunction, PiecewiseLinearFunction, SigmoidGradFunction

class SuperSpike(nn.Module):
    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return SuperSpikeFunction.apply(x, self.alpha)

class PiecewiseLinear(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return PiecewiseLinearFunction.apply(x, self.alpha)

class SigmoidSurrogate(nn.Module):
    def __init__(self, alpha=4.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return SigmoidGradFunction.apply(x, self.alpha)