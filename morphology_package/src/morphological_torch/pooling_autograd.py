import torch
import morph_cuda


class Pool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor, delta_provenance: torch.Tensor) -> tuple:
        provenance, kernel_size, stride, h, w, device = *ctx.saved_tensors, ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        dldf = morph_cuda.pool_backward_f(delta_up, provenance, kernel_size, stride, h, w, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, None, None, None


class MaxPool2DAutogradFunction(Pool2DAutogradFunction):

    @staticmethod
    def forward(ctx, input, kernel_size, stride, device):
        # Perform a forward pass.
        outputs, provenance = morph_cuda.maxpool_forward(input, kernel_size, stride, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(provenance)
        ctx.save_for_backward(provenance)
        ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device = kernel_size, stride, input.shape[2], input.shape[3], device
        return outputs, provenance


class MinPool2DAutogradFunction(Pool2DAutogradFunction):

    @staticmethod
    def forward(ctx, input, kernel_size, stride, device):
        # Perform a forward pass.
        outputs, provenance = morph_cuda.minpool_forward(input, kernel_size, stride, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(provenance)
        ctx.save_for_backward(provenance)
        ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device = kernel_size, stride, input.shape[2], input.shape[3], device
        return outputs, provenance


class ParameterizedMaxPool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, h, stride, device):
        # Perform a forward pass.
        outputs, provenance, poi_counter = morph_cuda.parameterized_maxpool_forward(input, h, stride, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(provenance)
        ctx.save_for_backward(provenance)
        ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device = h.shape[-1], stride, input.shape[2], input.shape[3], device
        return outputs, provenance, poi_counter

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor, delta_provenance: torch.Tensor, _) -> tuple:
        provenance, kernel_size, stride, h, w, device = *ctx.saved_tensors, ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        dldf = morph_cuda.pool_backward_f(delta_up, provenance, kernel_size, stride, h, w, device)
        dldh = morph_cuda.pool_backward_h(delta_up, provenance, kernel_size, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. stride and return indices bool.
        return dldf, dldh, None, None


class ParameterizedMinPool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, h, stride, device):
        # Perform a forward pass.
        outputs, provenance = morph_cuda.parameterized_minpool_forward(input, h, stride, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(provenance)
        ctx.save_for_backward(provenance)
        ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device = h.shape[-1], stride, input.shape[2], input.shape[3], device
        return outputs, provenance

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor, delta_provenance: torch.Tensor) -> tuple:
        provenance, kernel_size, stride, h, w, device = *ctx.saved_tensors, ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        dldf = morph_cuda.pool_backward_f(delta_up, provenance, kernel_size, stride, h, w, device)
        # Since the kernel is subtracted in minpool, the derivative must be multiplied by -1.
        dldh = - morph_cuda.pool_backward_h(delta_up, provenance, kernel_size, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. stride and return indices bool.
        return dldf, dldh, None, None
