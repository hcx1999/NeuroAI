import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

@torch.jit.script
def update_trace_lif(prev_trace: torch.Tensor, input_spike: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * prev_trace + input_spike

@torch.jit.script
def update_trace_alif(
    eps_a: torch.Tensor, 
    psi: torch.Tensor, 
    pre_trace: torch.Tensor, 
    rho: float, 
    beta: float
) -> torch.Tensor:
    """
    eps_a^{t+1} = (rho - psi * beta) * eps_a^t + psi * pre_trace
    """
    term1 = (rho - psi * beta) * eps_a
    term2 = psi * pre_trace
    return term1 + term2

@torch.jit.script
def compute_eligibility_alif(
    psi: torch.Tensor, 
    pre_trace: torch.Tensor, 
    eps_a: torch.Tensor, 
    beta: float
) -> torch.Tensor:
    """
    e_{ji}^t = psi^t * (pre_trace^{t-1} - beta * eps_a^t)  # Eq. 25 in paper
    """
    return psi * (pre_trace - beta * eps_a)

class BaseNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=1.0, tau_out=20.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.w_in = nn.Linear(input_size, hidden_size, bias=False)
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_out = nn.Linear(hidden_size, output_size, bias=False)
        self.kappa = torch.exp(torch.tensor(-dt / tau_out))

    def get_broadcast_matrix(self, broadcast_mode):
        if broadcast_mode == "symmetric":
            return self.w_out.weight
        elif broadcast_mode == "random":
            if not hasattr(self, 'B'):
                device = self.w_out.weight.device
                initial_B = torch.randn(self.w_out.out_features, self.hidden_size, device=device) / self.hidden_size ** 0.5
                self.register_buffer('B', initial_B)
            return self.B
        return None

class LIFNeuron(BaseNeuron):
    def __init__(self, input_size, hidden_size, output_size, dt=1.0, tau_m=20.0, v_th=1.0, **kwargs):
        super().__init__(input_size, hidden_size, output_size, dt=dt, **kwargs)
        self.alpha = torch.exp(torch.tensor(-dt / tau_m))
        self.v_th = v_th
        self.gamma = 0.3 

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device
        v = torch.zeros(batch_size, self.hidden_size, device=device)
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        
        trace_in = torch.zeros(batch_size, self.input_size, device=device)
        trace_rec = torch.zeros(batch_size, self.hidden_size, device=device)
        
        readout_seq = []
        readout = torch.zeros(batch_size, self.w_out.out_features, device=device)
        
        f_e_in = torch.zeros(batch_size, self.hidden_size, self.input_size, device=device)
        f_e_rec = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            v_prev = v.detach()
            z_prev = z.detach()
            i_t = self.w_in(x_t) + self.w_rec(z_prev)
            v_next = self.alpha * v_prev + i_t - z_prev * self.v_th
            
            relu_part = torch.relu(1.0 - torch.abs(v_next - self.v_th) / self.v_th)
            psi = (self.gamma / self.v_th) * relu_part
            z_next = (v_next > self.v_th).float()
            
            v_next = v_next * (1 - z_next) + (v_next - self.v_th) * z_next
            trace_in = update_trace_lif(trace_in, x_t, self.alpha)     # \bar{x}^t
            trace_rec = update_trace_lif(trace_rec, z_prev, self.alpha)     # \bar{z}^{t-1}
            
            e_in_t = psi.unsqueeze(2) * trace_in.unsqueeze(1)  # psi_j^t * \bar{x}_i^{t-1}
            e_rec_t = psi.unsqueeze(2) * trace_rec.unsqueeze(1)  # psi_j^t * \bar{z}_i^{t-1}
            
            f_e_in = self.kappa * f_e_in + e_in_t
            f_e_rec = self.kappa * f_e_rec + e_rec_t
            
            readout = self.kappa * readout + self.w_out(z_next)
            readout_seq.append(readout)
            
            v = v_next
            z = z_next

        readout_stack = torch.stack(readout_seq, dim=1)
        
        return readout_stack, (f_e_in, f_e_rec)

class ALIFNeuron(BaseNeuron):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dt: float = 1.0,
        tau_m: float = 20.0,
        tau_a: float = 2000.0,
        beta: float = 0.07,
        v_th_base: float = 0.6,
        gamma: float = 0.3,
        **kwargs,
    ):
        """
        Adaptive LIF neuron with configurable parameters via CLI.

        Parameters
        - tau_a: Adaptation time constant.
        - beta: Adaptation strength.
        - v_th_base: Base threshold voltage.
        - gamma: Surrogate gradient slope scaling.
        """
        super().__init__(input_size, hidden_size, output_size, dt=dt, **kwargs)
        self.alpha = torch.exp(torch.tensor(-dt / tau_m))
        self.rho = torch.exp(torch.tensor(-dt / tau_a))
        self.beta = beta
        self.v_th_base = v_th_base
        self.gamma = gamma

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device
        
        v = torch.zeros(batch_size, self.hidden_size, device=device)
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        a = torch.zeros(batch_size, self.hidden_size, device=device) # Adaptation variable
        
        trace_in = torch.zeros(batch_size, self.input_size, device=device)
        trace_rec = torch.zeros(batch_size, self.hidden_size, device=device)
        
        eps_a_in = torch.zeros(batch_size, self.hidden_size, self.input_size, device=device)
        eps_a_rec = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
        
        f_e_in = torch.zeros(batch_size, self.hidden_size, self.input_size, device=device)
        f_e_rec = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
        
        readout_seq = []
        readout = torch.zeros(batch_size, self.w_out.out_features, device=device)
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            a_prev = a.detach()
            v_prev = v.detach()
            z_prev = z.detach()
            A = self.v_th_base + self.beta * a_prev
            
            i_t = self.w_in(x_t) + self.w_rec(z_prev)
            v_next = self.alpha * v_prev + i_t - z_prev * self.v_th_base
            
            dist = v_next - A
            relu_part = torch.relu(1.0 - torch.abs(dist) / self.v_th_base)
            psi = (self.gamma / self.v_th_base) * relu_part
            
            z_next = (dist > 0).float()
            v_next = v_next * (1 - z_next) + (v_next - A) * z_next
            a_next = self.rho * a_prev + z_next
            
            trace_in = update_trace_lif(trace_in, x_t, self.alpha)
            trace_rec = update_trace_lif(trace_rec, z_prev, self.alpha)
            
            trace_in_exp = trace_in.unsqueeze(1)  # [B, 1, I]
            trace_rec_exp = trace_rec.unsqueeze(1)  # [B, 1, H]
            psi_exp = psi.unsqueeze(2)  # [B, H, 1]
            
            eps_a_in = update_trace_alif(eps_a_in, psi_exp, trace_in_exp, self.rho, self.beta)
            eps_a_rec = update_trace_alif(eps_a_rec, psi_exp, trace_rec_exp, self.rho, self.beta)
            
            e_in_t = compute_eligibility_alif(psi_exp, trace_in_exp, eps_a_in, self.beta)
            e_rec_t = compute_eligibility_alif(psi_exp, trace_rec_exp, eps_a_rec, self.beta)
            
            f_e_in = self.kappa * f_e_in + e_in_t
            f_e_rec = self.kappa * f_e_rec + e_rec_t
            
            readout = self.kappa * readout + self.w_out(z_next)
            readout_seq.append(readout)
            
            v = v_next
            z = z_next
            a = a_next
            
        return torch.stack(readout_seq, dim=1), (f_e_in, f_e_rec)

class RSNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        mode: str = "ALIF",
        broadcast: str = "random",
        alif_tau_a: float = 2000.0,
        alif_beta: float = 0.07,
        alif_v_th_base: float = 0.6,
        alif_gamma: float = 0.3,
    ):
        super().__init__()
        self.mode = mode
        if mode == "LIF":
            self.model = LIFNeuron(input_size, hidden_size, output_size)
        elif mode == "ALIF":
            self.model = ALIFNeuron(
                input_size,
                hidden_size,
                output_size,
                tau_a=alif_tau_a,
                beta=alif_beta,
                v_th_base=alif_v_th_base,
                gamma=alif_gamma,
            )
        self.broadcast = broadcast

    def forward(self, x):
        return self.model(x)

    def get_broadcast_matrix(self):
        return self.model.get_broadcast_matrix(self.broadcast)