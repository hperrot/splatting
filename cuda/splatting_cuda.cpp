#include <torch/extension.h>

// CUDA forward declarations

void splatting_forward_cuda_impl(
    const torch::Tensor frame,
    const torch::Tensor flow,
    torch::Tensor output
);

void splatting_backward_cuda_impl(
    const torch::Tensor frame,
    const torch::Tensor flow,
    const torch::Tensor grad_output,
    torch::Tensor grad_frame,
    torch::Tensor grad_flow
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x)  // ; CHECK_CONTIGUOUS(x)


void splatting_forward_cuda(
    const torch::Tensor frame,
    const torch::Tensor flow,
    torch::Tensor output
) {
    CHECK_INPUT(frame);
    CHECK_INPUT(flow);
    CHECK_INPUT(output);

    splatting_forward_cuda_impl(frame, flow, output);
}

void splatting_backward_cuda(
    const torch::Tensor frame,
    const torch::Tensor flow,
    const torch::Tensor grad_output,
    torch::Tensor grad_frame,
    torch::Tensor grad_flow
) {
    CHECK_INPUT(frame);
    CHECK_INPUT(flow);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_frame);
    CHECK_INPUT(grad_flow);

    splatting_backward_cuda_impl(frame, flow, grad_output, grad_frame, grad_flow);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splatting_forward_cuda", &splatting_forward_cuda, "splatting forward (CUDA)");
    m.def("splatting_backward_cuda", &splatting_backward_cuda, "splatting backward (CUDA)");
}

