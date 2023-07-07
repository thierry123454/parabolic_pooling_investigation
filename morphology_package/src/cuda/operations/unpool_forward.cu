#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
#include <cmath>


template <typename scalar_t>
__global__ void max_unpool_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int16_t,4,torch::RestrictPtrTraits> provenance,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    const int THREADS, const int STRIDE, const int K, const int W_OUT, const int PLANE) {
  // batch and c_out indicex.
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  // height and width indices.
  const int cell_idx = blockIdx.z * THREADS + threadIdx.x;
  if (cell_idx >= PLANE) return;
  const int h_out = (cell_idx / W_OUT);
  const int w_out = fmod(cell_idx, W_OUT);
  // In even-sized kernels the centre pixel is shifted compared to odd-sized kernels.
  const int k = K / 2;
  // Keep a record for the size of provenance, to never go outside bounds.
  const int PY = provenance.size(2);
  const int PX = provenance.size(3);
  // This is the (h_out, w_out) pixel location which we match to the provenance.
  int original_location = cell_idx;
  // Record the summed back-propped error as a double to deal with precision problems.
  double_t out_ = -100.0;
  // These variables are here to obtain the (h_out, w_out) pixel location
  // from the location of the provenance map and the value in the provenance map.
  int prov_i, prov_j, prov_ij, prov_ki, prov_kj, matched_location;
  for (int i=0; i < K; i += STRIDE) {
    for (int j=0; j < K; j += STRIDE) {
      // Determine which provenance location we should examine.
      // First, determine the h_in, w_in, then center around the kernel with (k / STRIDE).
      prov_i = (h_out + i) / STRIDE - (k / STRIDE);
      prov_j = (w_out + j) / STRIDE - (k / STRIDE);
      // If that goes out of bounds, continue.
      if (prov_i < 0 || prov_i >= PY || prov_j < 0 || prov_j >= PX) {
        continue;
      }
      // Read the provenance in the provenance map at that location.
      prov_ij = provenance[b][c][prov_i][prov_j];
      // This provenance value is kernel-centric, we go back to full spatial coordinates. Compute the offset.
      prov_ki = prov_ij / K - ((K % 2 == 1) ? k : 0);
      prov_kj = prov_ij % K - ((K % 2 == 1) ? k : 0);
      // Using the offset compute the actual spatial coordinates in (h_out, w_out).
      matched_location = ((prov_i * STRIDE) + prov_ki) * W_OUT + ((prov_j * STRIDE) + prov_kj);
      // And if these match our pixel in this thread, we save the delta_up value.
      // There is an issue here: If we were purely concerned with sampling, there
      // would be no case in which different values get placed back at the same
      // location. However, we transform the down-sampled feature volumes in
      // between, and must thus take a maximum in the case of max pool.
      // This complicates determining the derivative very much...
      if (matched_location == original_location) {
        out_ = std::max(out_, static_cast<double_t>(input[b][c][prov_i][prov_j]));
      }
    }
  }
  // Only write if necessary.
  if (out_ > -100.) {
    output[b][c][h_out][w_out] = static_cast<scalar_t>(out_);
  }
}

torch::Tensor max_unpool_forward_cuda(
    torch::Tensor input, torch::Tensor provenance, const int K, const int stride,
    const int H_OUT, const int W_OUT, const int device) {
  // Get the dimensions of the operation.
  const int B = input.size(0);
  const int C = input.size(1);
  const int H_IN = input.size(2);
  const int W_IN = input.size(3);
  const int PLANE_SIZE = H_OUT * W_OUT;

  // Initialize the output volume as a filled tensor.
  torch::Tensor output  = torch::full(torch::IntList{B, C, H_OUT, W_OUT}, -100, torch::dtype(torch::kF32).device(torch::kCUDA, device));

  const int THREADS = 192;
  const int Z = (H_OUT * W_OUT + THREADS - 1) / THREADS;
  const dim3 blocks(B, C, Z);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_unpool_forward_cuda", ([&] {
    max_unpool_forward_kernel<scalar_t><<<blocks, THREADS>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        provenance.packed_accessor32<int16_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        THREADS, stride, K, W_OUT, PLANE_SIZE);
  }));

  cudaDeviceSynchronize();
  return output;
}

template <typename scalar_t>
__global__ void min_unpool_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int16_t,4,torch::RestrictPtrTraits> provenance,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    const int THREADS, const int STRIDE, const int K, const int W_OUT, const int PLANE) {
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  const int cell_idx = blockIdx.z * THREADS + threadIdx.x;
  if (cell_idx >= PLANE) return;

  const int h_out = (cell_idx / W_OUT);
  const int w_out = fmod(cell_idx, W_OUT);
  const int k = K / 2;
  const int PY = provenance.size(2);
  const int PX = provenance.size(3);

  int original_location = cell_idx;
  double_t out_ = 100.0;
  int prov_i, prov_j, prov_ij, prov_ki, prov_kj, matched_location;
  for (int i=0; i < K; i += STRIDE) {
    for (int j=0; j < K; j += STRIDE) {
      prov_i = (h_out + i) / STRIDE - (k / STRIDE);
      prov_j = (w_out + j) / STRIDE - (k / STRIDE);
      if (prov_i < 0 || prov_i >= PY || prov_j < 0 || prov_j >= PX) {
        continue;
      }
      prov_ij = provenance[b][c][prov_i][prov_j];
      prov_ki = prov_ij / K - ((K % 2 == 1) ? k : 0);
      prov_kj = prov_ij % K - ((K % 2 == 1) ? k : 0);
      matched_location = ((prov_i * STRIDE) + prov_ki) * W_OUT + ((prov_j * STRIDE) + prov_kj);
      if (matched_location == original_location) {
        out_ = std::min(out_, static_cast<double_t>(input[b][c][prov_i][prov_j]));
      }
    }
  }
  if (out_ < 100.) {
    output[b][c][h_out][w_out] = static_cast<scalar_t>(out_);
  }
}

torch::Tensor min_unpool_forward_cuda(
    torch::Tensor input, torch::Tensor provenance, const int K, const int stride,
    const int H_OUT, const int W_OUT, const int device) {
  // Get the dimensions of the operation.
  const int B = input.size(0);
  const int C = input.size(1);
  const int H_IN = input.size(2);
  const int W_IN = input.size(3);
  const int PLANE_SIZE = H_OUT * W_OUT;

  // Initialize the output volume as a filled tensor.
  torch::Tensor output  = torch::full(torch::IntList{B, C, H_OUT, W_OUT}, 100, torch::dtype(torch::kF32).device(torch::kCUDA, device));

  const int THREADS = 192;
  const int Z = (H_OUT * W_OUT + THREADS - 1) / THREADS;
  const dim3 blocks(B, C, Z);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_unpool_forward_cuda", ([&] {
    min_unpool_forward_kernel<scalar_t><<<blocks, THREADS>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        provenance.packed_accessor32<int16_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        THREADS, stride, K, W_OUT, PLANE_SIZE);
  }));

  cudaDeviceSynchronize();
  return output;
}
