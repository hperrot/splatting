#include <torch/extension.h>
#include <cmath>


template<typename scalar_t>
void splatting_forward_cpu_impl(
    const torch::Tensor frame,
    const torch::Tensor flow,
    torch::Tensor output
) {
    const auto frame_a = frame.accessor<scalar_t, 4>();
    const auto flow_a = flow.accessor<scalar_t, 4>();
    auto output_a = output.accessor<scalar_t, 4>();

    const auto N = frame.size(0);
    const auto C = frame.size(1);
    const auto Y = frame.size(2);
    const auto X = frame.size(3);

    for (auto n=0; n<N; n++){
        for (auto c=0; c<C; c++) {
            for(auto y=0; y<Y; y++) {
                for(auto x=0; x<X; x++) {

                    const scalar_t outputX = (scalar_t) x + flow_a[n][0][y][x];
                    const scalar_t outputY = (scalar_t) y + flow_a[n][1][y][x];

                    const auto northwestX = (int) (floor(outputX));
                    const auto northwestY = (int) (floor(outputY));
                    const auto northeastX = northwestX + 1;
                    const auto northeastY = northwestY;
                    const auto southwestX = northwestX;
                    const auto southwestY = northwestY + 1;
                    const auto southeastX = northwestX + 1;
                    const auto southeastY = northwestY + 1;

                    const scalar_t northwest = ((scalar_t) (southeastX) - outputX   ) * ((scalar_t) (southeastY) - outputY   );
                    const scalar_t northeast = (outputX    - (scalar_t) (southwestX)) * ((scalar_t) (southwestY) - outputY   );
                    const scalar_t southwest = ((scalar_t) (northeastX) - outputX   ) * (outputY    - (scalar_t) (northeastY));
                    const scalar_t southeast = (outputX    - (scalar_t) (northwestX)) * (outputY    - (scalar_t) (northwestY));

                    if ((northwestX >= 0) && (northwestX < X) && (northwestY >= 0) && (northwestY < Y)) {
                        output_a[n][c][northwestY][northwestX] += northwest * frame_a[n][c][y][x];
                    }
                    if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) && (northeastY < Y)) {
                        output_a[n][c][northeastY][northeastX] += northeast * frame_a[n][c][y][x];
                    }
                    if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) && (southwestY < Y)) {
                        output_a[n][c][southwestY][southwestX] += southwest * frame_a[n][c][y][x];
                    }
                    if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) && (southeastY < Y)) {
                        output_a[n][c][southeastY][southeastX] += southeast * frame_a[n][c][y][x];
                    }

                }
            }
        }
    }
}


template<typename scalar_t>
void splatting_backward_cpu_impl(
    const torch::Tensor frame,
    const torch::Tensor flow,
    const torch::Tensor grad_output,
    torch::Tensor grad_frame,
    torch::Tensor grad_flow
) {
    const auto frame_a = frame.accessor<scalar_t, 4>();
    const auto flow_a = flow.accessor<scalar_t, 4>();
    const auto grad_output_a = grad_output.accessor<scalar_t, 4>();
    auto grad_frame_a = grad_frame.accessor<scalar_t, 4>();
    auto grad_flow_a = grad_flow.accessor<scalar_t, 4>();

    const auto N = frame.size(0);
    const auto C = frame.size(1);
    const auto Y = frame.size(2);
    const auto X = frame.size(3);

    for (auto n=0; n<N; n++){
        for (auto c=0; c<C; c++) {
            for(auto y=0; y<Y; y++) {
                for(auto x=0; x<X; x++) {

                    const scalar_t outputX = (scalar_t) x + flow_a[n][0][y][x];
                    const scalar_t outputY = (scalar_t) y + flow_a[n][1][y][x];

                    const auto northwestX = (int) (floor(outputX));
                    const auto northwestY = (int) (floor(outputY));
                    const auto northeastX = northwestX + 1;
                    const auto northeastY = northwestY;
                    const auto southwestX = northwestX;
                    const auto southwestY = northwestY + 1;
                    const auto southeastX = northwestX + 1;
                    const auto southeastY = northwestY + 1;

                    const scalar_t northwest = ((scalar_t) (southeastX) - outputX   ) * ((scalar_t) (southeastY) - outputY   );
                    const scalar_t northeast = (outputX    - (scalar_t) (southwestX)) * ((scalar_t) (southwestY) - outputY   );
                    const scalar_t southwest = ((scalar_t) (northeastX) - outputX   ) * (outputY    - (scalar_t) (northeastY));
                    const scalar_t southeast = (outputX    - (scalar_t) (northwestX)) * (outputY    - (scalar_t) (northwestY));

                    const scalar_t frame_here = frame_a[n][c][y][x];

                    if ((northwestX >= 0) && (northwestX < X) && (northwestY >= 0) && (northwestY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][northwestY][northwestX];
                        grad_frame_a[n][c][y][x] += northwest * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((scalar_t) (-1.0)) * ((scalar_t) (southeastY) - outputY   );
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * ((scalar_t) (southeastX) - outputX   ) * ((scalar_t) (-1.0));
                    }
                    if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) && (northeastY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][northeastY][northeastX];
                        grad_frame_a[n][c][y][x] += northeast * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((scalar_t) (+1.0)) * ((scalar_t) (southwestY) - outputY   );
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * (outputX    - (scalar_t) (southwestX)) * ((scalar_t) (-1.0));
                    }
                    if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) && (southwestY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][southwestY][southwestX];
                        grad_frame_a[n][c][y][x] += southwest * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((scalar_t) (-1.0)) * (outputY    - (scalar_t) (northeastY));
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * ((scalar_t) (northeastX) - outputX   ) * ((scalar_t) (+1.0));
                    }
                    if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) && (southeastY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][southeastY][southeastX];
                        grad_frame_a[n][c][y][x] += southeast * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((scalar_t) (+1.0)) * (outputY    - (scalar_t) (northwestY));
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * (outputX    - (scalar_t) (northwestX)) * ((scalar_t) (+1.0));
                    }

                }
            }
        }
    }

}


void splatting_forward(
    const torch::Tensor frame,
    const torch::Tensor flow,
    torch::Tensor output
) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        frame.scalar_type(),
        "splatting_forward_cpu",
        [&] {
            splatting_forward_cpu_impl<scalar_t>(frame, flow, output);
        }
    );
}


void splatting_backward(
    const torch::Tensor frame,
    const torch::Tensor flow,
    const torch::Tensor grad_output,
    torch::Tensor grad_frame,
    torch::Tensor grad_flow
) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        frame.scalar_type(),
        "splatting_backward_cpu",
        [&] {
            splatting_backward_cpu_impl<scalar_t>(frame, flow, grad_output, grad_frame, grad_flow);
        }
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splatting_forward", &splatting_forward, "splatting forward");
    m.def("splatting_backward", &splatting_backward, "splatting backward");
}
