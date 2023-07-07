import torch
import morphological_cuda


class CudaDilation2D(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, device: int = 0) -> None:
        super(CudaDilation2D, self).__init__()
        self.ins = in_channels
        self.outs = out_channels
        assert kernel_size % 2 == 1, "Kernel size has to be odd."
        self.ks = kernel_size
        # Initialize the parameters.
        h = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        torch.nn.init.xavier_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)
        self.device = device

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return Dilation2DAutogradFunction.apply(f, self.h, self.device)

    def to(self, device_reference):
        """ Set the device reference to the correct GPU.
        """
        super().to(device_reference)
        if type(device_reference) == torch.device:
            self.device = device_reference.index
        return self

    def extra_repr(self) -> str:
        return f'in_channels={self.ins}, out_channels={self.outs}, kernel_size={self.ks}'


class Dilation2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights, device):
        # Register shapes.
        _, c_in, _, k = weights.shape
        # Perform a forward pass.
        outputs, provenance = morphological_cuda.dilation_forward(input, weights, device)
        ctx.save_for_backward(provenance)
        ctx.c_in, ctx.k, ctx.device = c_in, k, device
        return outputs

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        provenance, c_in, k, device = *ctx.saved_tensors, ctx.c_in, ctx.k, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs and parameters.
        delta_up = delta_up.contiguous()
        # import time
        # t1 = time.time()
        dldf = morphological_cuda.dilation_backward_f(delta_up, provenance, c_in, k, device)
        # torch.cuda.synchronize()
        # t2 = time.time()
        dldh = morphological_cuda.dilation_backward_h(delta_up, provenance, c_in, k, device)
        # torch.cuda.synchronize()
        # t3 = time.time()
        # print(t2 - t1, t3 - t2)
        # Return the gradients w.r.t. the input signal, the parameters h, and None to the kernel size and utils.
        return dldf, dldh
