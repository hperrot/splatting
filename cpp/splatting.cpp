#include <torch/extension.h>

torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("d_sigmoid", &d_sigmoid, "d sigmoid");
}
