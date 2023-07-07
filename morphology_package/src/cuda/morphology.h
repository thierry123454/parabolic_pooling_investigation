#include <torch/extension.h>
#include <vector>
#include <iostream>


// -- Pooling operations --
// - forward operations -
std::vector<torch::Tensor> maxpool_forward_cuda(
    torch::Tensor input, const int kernel_size, const int stride, const int device
);

std::vector<torch::Tensor> maxpool_double_forward_cuda(
    torch::Tensor input, const int kernel_size, const int stride, const int device
);

std::vector<torch::Tensor> minpool_forward_cuda(
    torch::Tensor input, const int kernel_size, const int stride, const int device
);

std::vector<torch::Tensor> minpool_double_forward_cuda(
    torch::Tensor input, const int kernel_size, const int stride, const int device
);

std::vector<torch::Tensor> parameterized_maxpool_forward_cuda(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device
);

std::vector<torch::Tensor> parameterized_maxpool_double_forward_cuda(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device
);

std::vector<torch::Tensor> parameterized_minpool_forward_cuda(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device
);

std::vector<torch::Tensor> parameterized_minpool_double_forward_cuda(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device
);

// - backward operations -
torch::Tensor pool_backward_f_cuda(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int stride, const int H_OUT, const int W_OUT, const int device
);

torch::Tensor pool_backward_h_cuda(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int device
);

// -- Unpooling operations --
// - forward operations -
torch::Tensor max_unpool_forward_cuda(
    torch::Tensor input, torch::Tensor provenance, const int K, const int stride, const int H_OUT, const int W_OUT, const int device
);

torch::Tensor min_unpool_forward_cuda(
    torch::Tensor input, torch::Tensor provenance, const int K, const int stride, const int H_OUT, const int W_OUT, const int device
);

// - backward operations -
torch::Tensor unpool_backward_cuda(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int stride, const int device
);
