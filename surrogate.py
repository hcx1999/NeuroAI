import torch

class SuperSpikeFunction(torch.autograd.Function):
    '''
    F. Zenke and T. P. Vogels, 
    "The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks,"
    in Neural Computation, vol. 33, no. 4, pp. 899-925, 26 March 2021
    https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
    '''
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input)
        ctx.beta = beta
        out = input.gt(0).float() # Heaviside step function
        return out

    @staticmethod
    def backward(ctx, grad_output):
        '''
        sigma'(x) = 1 / (beta * |x| + 1)^2
        '''
        input, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output.clone()
        sgax = (input.abs() * beta) + 1
        grad = grad_input / (sgax * sgax)
        return grad, None

class AsymSuperSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input)
        ctx.beta = beta
        out = input.gt(0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        '''
        sigma'(x) = beta / (beta * |x| + 1)^2
        '''
        input, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output.clone()
        sgax = (input.abs() * beta) + 1
        grad = grad_input * beta / (sgax * sgax)
        return grad, None

class PiecewiseLinearFunction(torch.autograd.Function):
    ''' 
    Esser et al., "Convolutional Networks for Fast, Energy-Efficient Neuromorphic Computing,"
    in Proceedings of the National Academy of Sciences, vol. 113, no. 41, pp. 11441-11446, 10 October 2016.
    '''
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input)
        ctx.beta = beta
        out = input.gt(0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        '''
        sigma'(x) = max(0, 1 - beta * |x|)
        '''
        input, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output.clone()
        grad_shape = torch.clamp(1.0 - beta * input.abs(), min=0.0)
        grad = grad_input * grad_shape
        return grad, None

class SigmoidGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input)
        ctx.beta = beta
        out = input.gt(0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        \sigma'(x) = sigma(x) * (1 - sigma(x))
        sigma(x) = 1 / (1 + exp(-beta * x))
        """
        input, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output.clone()
        sgax = torch.sigmoid(input * beta)
        grad = grad_input * (1.0 - sgax) * sgax * beta
        return grad, None