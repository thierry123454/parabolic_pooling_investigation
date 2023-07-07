#include "morphology.h"
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>


// -- Pooling operations --
// - forward operations -
std::vector<torch::Tensor> maxpool_forward(
    torch::Tensor input, const int kernel_size, const int stride, const int device) {
  CHECK_INPUT(input);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return maxpool_forward_cuda(input, kernel_size, stride, device);
}

std::vector<torch::Tensor> maxpool_double_forward(
    torch::Tensor input, const int kernel_size, const int stride, const int device) {
  CHECK_INPUT(input);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return maxpool_double_forward_cuda(input, kernel_size, stride, device);
}

std::vector<torch::Tensor> minpool_forward(
    torch::Tensor input, const int kernel_size, const int stride, const int device) {
  CHECK_INPUT(input);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return minpool_forward_cuda(input, kernel_size, stride, device);
}

std::vector<torch::Tensor> minpool_double_forward(
    torch::Tensor input, const int kernel_size, const int stride, const int device) {
  CHECK_INPUT(input);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return minpool_double_forward_cuda(input, kernel_size, stride, device);
}

std::vector<torch::Tensor> parameterized_maxpool_forward(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return parameterized_maxpool_forward_cuda(input, weights, stride, device);
}

std::vector<torch::Tensor> parameterized_maxpool_double_forward(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return parameterized_maxpool_double_forward_cuda(input, weights, stride, device);
}

std::vector<torch::Tensor> parameterized_minpool_forward(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return parameterized_minpool_forward_cuda(input, weights, stride, device);
}

std::vector<torch::Tensor> parameterized_minpool_double_forward(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return parameterized_minpool_double_forward_cuda(input, weights, stride, device);
}

// - backward operations -
torch::Tensor pool_backward_f(
    torch::Tensor delta_up, torch::Tensor provenance,
    const int kernel_size, const int stride, const int H_OUT,
    const int W_OUT, const int device) {
  CHECK_INPUT(delta_up);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta_up));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(delta_up));
  return pool_backward_f_cuda(delta_up, provenance, kernel_size, stride, H_OUT, W_OUT, device);
}

torch::Tensor pool_backward_h(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int device) {
  CHECK_INPUT(delta_up);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta_up));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return pool_backward_h_cuda(delta_up, provenance, K, device);
}

// -- Unpooling operations --
// - forward operations -
torch::Tensor max_unpool_forward(
  torch::Tensor input, torch::Tensor provenance,
  const int kernel_size, const int stride, const int H_OUT, const int W_OUT, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return max_unpool_forward_cuda(input, provenance, kernel_size, stride, H_OUT, W_OUT, device);
}

torch::Tensor min_unpool_forward(
  torch::Tensor input, torch::Tensor provenance,
  const int kernel_size, const int stride, const int H_OUT, const int W_OUT, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return min_unpool_forward_cuda(input, provenance, kernel_size, stride, H_OUT, W_OUT, device);
}

// - backward operations -
torch::Tensor unpool_backward(
  torch::Tensor delta_up, torch::Tensor provenance, const int kernel_size,
  const int stride, const int device) {
  CHECK_INPUT(delta_up);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta_up));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return unpool_backward_cuda(delta_up, provenance, kernel_size, stride, device);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("maxpool_forward", &maxpool_forward, "Standard (unparameterized) forward maxpool (CUDA).");
  m.def("maxpool_double_forward", &maxpool_double_forward, "Perform (unparameterized) forward maxpool twice (CUDA).");
  m.def("minpool_forward", &minpool_forward, "Standard (unparameterized) forward minpool (CUDA).");
  m.def("minpool_double_forward", &minpool_double_forward, "Perform (unparameterized) forward minpool twice (CUDA).");
  m.def("parameterized_maxpool_forward", &parameterized_maxpool_forward, "Parameterized maxpool forward (CUDA).");
  m.def("parameterized_maxpool_double_forward", &parameterized_maxpool_double_forward, "Parameterized forward maxpool twice (CUDA).");
  m.def("parameterized_minpool_forward", &parameterized_minpool_forward, "Parameterized minpool forward (CUDA).");
  m.def("parameterized_minpool_double_forward", &parameterized_minpool_double_forward, "Parameterized forward minpool twice (CUDA).");
  m.def("pool_backward_f", &pool_backward_f, "Backward pool, both for min and max pools, with respect to the signal (CUDA).");
  m.def("pool_backward_h", &pool_backward_h, "Backward pool, both for min and max pools, with respect to the kernel (CUDA).");
  m.def("max_unpool_forward", &max_unpool_forward, "(unparameterized) forward max unpool operation (CUDA).");
  m.def("min_unpool_forward", &min_unpool_forward, "(unparameterized) forward min unpool operation (CUDA).");
  m.def("unpool_backward", &unpool_backward, "backward unpool operation (CUDA)");
}
