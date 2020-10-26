#include <torch/extension.h>

#include <ATen/cuda/detail/KernelUtils.h> // for CUDA_KERNEL_LOOP and GET_BLOCKS
#include <cuda.h>
#include <cuda_runtime.h>
#include <THC/THCAtomics.cuh> // for gpuAtomicAdd

using namespace at::cuda::detail;

template <typename scalar_t>
__global__ void splatting_forward_cuda_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        frame_a,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        flow_a,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        output_a) {
  const size_t N = frame_a.size(0);
  const size_t C = frame_a.size(1);
  const size_t Y = frame_a.size(2);
  const size_t X = frame_a.size(3);
  const size_t total_step = N * C * Y * X;

  CUDA_KERNEL_LOOP(index, total_step) {
    const size_t y = index % Y;
    const size_t x = (index / Y) % X;
    const size_t c = (index / (X * Y) % C);
    const size_t n = index / (C * X * Y);

    const scalar_t outputX = (scalar_t)x + flow_a[n][0][y][x];
    const scalar_t outputY = (scalar_t)y + flow_a[n][1][y][x];

    const int64_t northwestX = (int64_t)(floor(outputX));
    const int64_t northwestY = (int64_t)(floor(outputY));
    const int64_t northeastX = northwestX + 1;
    const int64_t northeastY = northwestY;
    const int64_t southwestX = northwestX;
    const int64_t southwestY = northwestY + 1;
    const int64_t southeastX = northwestX + 1;
    const int64_t southeastY = northwestY + 1;

    const scalar_t northwest =
        ((scalar_t)(southeastX)-outputX) * ((scalar_t)(southeastY)-outputY);
    const scalar_t northeast =
        (outputX - (scalar_t)(southwestX)) * ((scalar_t)(southwestY)-outputY);
    const scalar_t southwest =
        ((scalar_t)(northeastX)-outputX) * (outputY - (scalar_t)(northeastY));
    const scalar_t southeast =
        (outputX - (scalar_t)(northwestX)) * (outputY - (scalar_t)(northwestY));

    if ((northwestX >= 0) && (northwestX < X) && (northwestY >= 0) &&
        (northwestY < Y)) {
      gpuAtomicAdd(
          &(output_a[n][c][northwestY][northwestX]),
          northwest * frame_a[n][c][y][x]);
    }
    if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) &&
        (northeastY < Y)) {
      gpuAtomicAdd(
          &(output_a[n][c][northeastY][northeastX]),
          northeast * frame_a[n][c][y][x]);
    }
    if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) &&
        (southwestY < Y)) {
      gpuAtomicAdd(
          &(output_a[n][c][southwestY][southwestX]),
          southwest * frame_a[n][c][y][x]);
    }
    if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) &&
        (southeastY < Y)) {
      gpuAtomicAdd(
          &(output_a[n][c][southeastY][southeastX]),
          southeast * frame_a[n][c][y][x]);
    }
  }
}

template <typename scalar_t>
__global__ void splatting_backward_cuda_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        frame_a,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        flow_a,
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        grad_output_a,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        grad_frame_a,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        grad_flow_a) {
  const size_t N = frame_a.size(0);
  const size_t C = frame_a.size(1);
  const size_t Y = frame_a.size(2);
  const size_t X = frame_a.size(3);
  const size_t total_step = N * C * Y * X;

  CUDA_KERNEL_LOOP(index, total_step) {
    const size_t y = index % Y;
    const size_t x = (index / Y) % X;
    const size_t c = (index / (X * Y) % C);
    const size_t n = index / (C * X * Y);

    const scalar_t outputX = (scalar_t)x + flow_a[n][0][y][x];
    const scalar_t outputY = (scalar_t)y + flow_a[n][1][y][x];

    const int64_t northwestX = (int64_t)(floor(outputX));
    const int64_t northwestY = (int64_t)(floor(outputY));
    const int64_t northeastX = northwestX + 1;
    const int64_t northeastY = northwestY;
    const int64_t southwestX = northwestX;
    const int64_t southwestY = northwestY + 1;
    const int64_t southeastX = northwestX + 1;
    const int64_t southeastY = northwestY + 1;

    const scalar_t northwest =
        ((scalar_t)(southeastX)-outputX) * ((scalar_t)(southeastY)-outputY);
    const scalar_t northeast =
        (outputX - (scalar_t)(southwestX)) * ((scalar_t)(southwestY)-outputY);
    const scalar_t southwest =
        ((scalar_t)(northeastX)-outputX) * (outputY - (scalar_t)(northeastY));
    const scalar_t southeast =
        (outputX - (scalar_t)(northwestX)) * (outputY - (scalar_t)(northwestY));

    const scalar_t frame_here = frame_a[n][c][y][x];

    if ((northwestX >= 0) && (northwestX < X) && (northwestY >= 0) &&
        (northwestY < Y)) {
      const scalar_t grad_output_here =
          grad_output_a[n][c][northwestY][northwestX];
      gpuAtomicAdd(&(grad_frame_a[n][c][y][x]), northwest * grad_output_here);
      gpuAtomicAdd(
          &(grad_flow_a[n][0][y][x]),
          frame_here * grad_output_here * ((scalar_t)(-1.0)) *
              ((scalar_t)(southeastY)-outputY));
      gpuAtomicAdd(
          &(grad_flow_a[n][1][y][x]),
          frame_here * grad_output_here * ((scalar_t)(southeastX)-outputX) *
              ((scalar_t)(-1.0)));
    }
    if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) &&
        (northeastY < Y)) {
      const scalar_t grad_output_here =
          grad_output_a[n][c][northeastY][northeastX];
      gpuAtomicAdd(&(grad_frame_a[n][c][y][x]), northeast * grad_output_here);
      gpuAtomicAdd(
          &(grad_flow_a[n][0][y][x]),
          frame_here * grad_output_here * ((scalar_t)(+1.0)) *
              ((scalar_t)(southwestY)-outputY));
      gpuAtomicAdd(
          &(grad_flow_a[n][1][y][x]),
          frame_here * grad_output_here * (outputX - (scalar_t)(southwestX)) *
              ((scalar_t)(-1.0)));
    }
    if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) &&
        (southwestY < Y)) {
      const scalar_t grad_output_here =
          grad_output_a[n][c][southwestY][southwestX];
      gpuAtomicAdd(&(grad_frame_a[n][c][y][x]), southwest * grad_output_here);
      gpuAtomicAdd(
          &(grad_flow_a[n][0][y][x]),
          frame_here * grad_output_here * ((scalar_t)(-1.0)) *
              (outputY - (scalar_t)(northeastY)));
      gpuAtomicAdd(
          &(grad_flow_a[n][1][y][x]),
          frame_here * grad_output_here * ((scalar_t)(northeastX)-outputX) *
              ((scalar_t)(+1.0)));
    }
    if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) &&
        (southeastY < Y)) {
      const scalar_t grad_output_here =
          grad_output_a[n][c][southeastY][southeastX];
      gpuAtomicAdd(&(grad_frame_a[n][c][y][x]), southeast * grad_output_here);
      gpuAtomicAdd(
          &(grad_flow_a[n][0][y][x]),
          frame_here * grad_output_here * ((scalar_t)(+1.0)) *
              (outputY - (scalar_t)(northwestY)));
      gpuAtomicAdd(
          &(grad_flow_a[n][1][y][x]),
          frame_here * grad_output_here * (outputX - (scalar_t)(northwestX)) *
              ((scalar_t)(+1.0)));
    }
  }
}

void splatting_forward_cuda_impl(
    const torch::Tensor frame,
    const torch::Tensor flow,
    torch::Tensor output) {
  const size_t num_steps = frame.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      frame.type(), "splatting_forward_cuda", [&] {
        splatting_forward_cuda_kernel<
            scalar_t><<<GET_BLOCKS(num_steps), CUDA_NUM_THREADS>>>(
            frame.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            flow.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>());
      });
}

void splatting_backward_cuda_impl(
    const torch::Tensor frame,
    const torch::Tensor flow,
    const torch::Tensor grad_output,
    torch::Tensor grad_frame,
    torch::Tensor grad_flow) {
  const size_t num_steps = frame.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      frame.type(), "splatting_backward_cuda", [&] {
        splatting_backward_cuda_kernel<
            scalar_t><<<GET_BLOCKS(num_steps), CUDA_NUM_THREADS>>>(
            frame.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            flow.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_output
                .packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_frame
                .packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_flow
                .packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>());
      });
}
