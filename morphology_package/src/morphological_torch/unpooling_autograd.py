import torch
import morph_cuda


class SparseUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, pool_provenance, size, in_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morph_cuda.max_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Because of the way the CUDA code is written, we must set the INF values back to zero.
        upsampled_ins = torch.where(upsampled_ins < torch.tensor([-99.], dtype=torch.float32).cuda(),
                                    torch.tensor([0.], dtype=torch.float32).cuda(), upsampled_ins)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance)
        ctx.save_for_backward(pool_provenance)
        ctx.in_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, stride, h, w, device
        return upsampled_ins

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, in_ks, stride, h, w, device = *ctx.saved_tensors, ctx.in_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Compute delta_up it w.r.t. the up-sampling operation.
        dldf = morph_cuda.unpool_backward(delta_up.contiguous(), error_provenance, in_ks, stride, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, None, None, None, None, None


class Unpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, backward_provenance = ctx.saved_tensors
        in_ks, out_ks, stride, h, w, device = ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        # First compute the gradients w.r.t. the pooling operation.
        dldf_pooled = morph_cuda.pool_backward_f(delta_up, backward_provenance, out_ks, 1, h, w, device)
        # Then compute it w.r.t. the up-sampling operation.
        dldf = morph_cuda.unpool_backward(dldf_pooled, error_provenance, in_ks, stride, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, None, None, None, None, None, None


class MaxUnpool2DAutogradFunction(Unpool2DAutogradFunction):

    @staticmethod
    def forward(ctx, inputs, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morph_cuda.max_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Then perform a max pool with the desired out_ks parameters, at stride 1.
        outputs, backward_provenance = morph_cuda.maxpool_forward(upsampled_ins, out_ks, 1, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs


class MinUnpool2DAutogradFunction(Unpool2DAutogradFunction):

    @staticmethod
    def forward(ctx, inputs, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        upsampled_ins = morph_cuda.min_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        outputs, backward_provenance = morph_cuda.minpool_forward(upsampled_ins, out_ks, 1, device)
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs


class DoubleMaxUnpool2DAutogradFunction(Unpool2DAutogradFunction):

    @staticmethod
    def forward(ctx, inputs, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        upsampled_ins = morph_cuda.max_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # In this case we perform the pooling twice, to deal with the sparsity issue.
        outputs, backward_provenance = morph_cuda.maxpool_double_forward(upsampled_ins, out_ks, 1, device)
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs


class DoubleMinUnpool2DAutogradFunction(Unpool2DAutogradFunction):

    @staticmethod
    def forward(ctx, inputs, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        upsampled_ins = morph_cuda.min_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        outputs, backward_provenance = morph_cuda.minpool_double_forward(upsampled_ins, out_ks, 1, device)
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs


class ParameterizedMaxUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morph_cuda.max_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Then perform a max pool with the desired out_ks parameters, at stride 1.
        outputs, backward_provenance = morph_cuda.parameterized_maxpool_forward(upsampled_ins, weights, 1, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, backward_provenance = ctx.saved_tensors
        in_ks, out_ks, stride, h, w, device = ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        # First compute the gradients w.r.t. the pooling operation.
        dldf_pooled = morph_cuda.pool_backward_f(delta_up, backward_provenance, out_ks, 1, h, w, device)
        # Then compute it w.r.t. the up-sampling operation.
        dldf = morph_cuda.unpool_backward(dldf_pooled, error_provenance, in_ks, stride, device)
        # Also compute the derivative w.r.t. the weights.
        dldh = morph_cuda.pool_backward_h(delta_up, backward_provenance, out_ks, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, dldh, None, None, None, None, None, None


class ParameterizedDoubleMaxUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morph_cuda.max_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Then perform a max pool with the desired out_ks parameters, at stride 1.
        outputs, backward_provenance = morph_cuda.parameterized_maxpool_double_forward(upsampled_ins, weights, 1, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, backward_provenance = ctx.saved_tensors
        in_ks, out_ks, stride, h, w, device = ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        # First compute the gradients w.r.t. the pooling operation.
        dldf_pooled = morph_cuda.pool_backward_f(delta_up, backward_provenance, out_ks, 1, h, w, device)
        # Then compute it w.r.t. the up-sampling operation.
        dldf = morph_cuda.unpool_backward(dldf_pooled, error_provenance, in_ks, stride, device)
        # Also compute the derivative w.r.t. the weights.
        dldh = morph_cuda.pool_backward_h(delta_up, backward_provenance, out_ks, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, dldh, None, None, None, None, None, None


class ParameterizedMinUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morph_cuda.min_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Then perform a max pool with the desired out_ks parameters, at stride 1.
        outputs, backward_provenance = morph_cuda.parameterized_minpool_forward(upsampled_ins, weights, 1, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, backward_provenance = ctx.saved_tensors
        in_ks, out_ks, stride, h, w, device = ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        # First compute the gradients w.r.t. the pooling operation.
        dldf_pooled = morph_cuda.pool_backward_f(delta_up, backward_provenance, out_ks, 1, h, w, device)
        # Then compute it w.r.t. the up-sampling operation.
        dldf = morph_cuda.unpool_backward(dldf_pooled, error_provenance, in_ks, stride, device)
        # Also compute the derivative w.r.t. the weights.
        dldh = - morph_cuda.pool_backward_h(delta_up, backward_provenance, out_ks, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, dldh, None, None, None, None, None, None


class ParameterizedDoubleMinUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morph_cuda.min_unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Then perform a max pool with the desired out_ks parameters, at stride 1.
        outputs, backward_provenance = morph_cuda.parameterized_minpool_double_forward(upsampled_ins, weights, 1, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, backward_provenance = ctx.saved_tensors
        in_ks, out_ks, stride, h, w, device = ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        # First compute the gradients w.r.t. the pooling operation.
        dldf_pooled = morph_cuda.pool_backward_f(delta_up, backward_provenance, out_ks, 1, h, w, device)
        # Then compute it w.r.t. the up-sampling operation.
        dldf = morph_cuda.unpool_backward(dldf_pooled, error_provenance, in_ks, stride, device)
        # Also compute the derivative w.r.t. the weights.
        dldh = - morph_cuda.pool_backward_h(delta_up, backward_provenance, out_ks, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, dldh, None, None, None, None, None, None
