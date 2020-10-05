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

                    const float outputX = (float) x + flow_a[n][0][y][x];
                    const float outputY = (float) y + flow_a[n][1][y][x];

                    const auto northwestX = (int) (floor(outputX));
                    const auto northwestY = (int) (floor(outputY));
                    const auto northeastX = northwestX + 1;
                    const auto northeastY = northwestY;
                    const auto southwestX = northwestX;
                    const auto southwestY = northwestY + 1;
                    const auto southeastX = northwestX + 1;
                    const auto southeastY = northwestY + 1;

                    const auto northwest = ((float) (southeastX) - outputX   ) * ((float) (southeastY) - outputY   );
                    const auto northeast = (outputX    - (float) (southwestX)) * ((float) (southwestY) - outputY   );
                    const auto southwest = ((float) (northeastX) - outputX   ) * (outputY    - (float) (northeastY));
                    const auto southeast = (outputX    - (float) (northwestX)) * (outputY    - (float) (northwestY));

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

                    const float outputX = (float) x + flow_a[n][0][y][x];
                    const float outputY = (float) y + flow_a[n][1][y][x];

                    const auto northwestX = (int) (floor(outputX));
                    const auto northwestY = (int) (floor(outputY));
                    const auto northeastX = northwestX + 1;
                    const auto northeastY = northwestY;
                    const auto southwestX = northwestX;
                    const auto southwestY = northwestY + 1;
                    const auto southeastX = northwestX + 1;
                    const auto southeastY = northwestY + 1;

                    const auto northwest = ((float) (southeastX) - outputX   ) * ((float) (southeastY) - outputY   );
                    const auto northeast = (outputX    - (float) (southwestX)) * ((float) (southwestY) - outputY   );
                    const auto southwest = ((float) (northeastX) - outputX   ) * (outputY    - (float) (northeastY));
                    const auto southeast = (outputX    - (float) (northwestX)) * (outputY    - (float) (northwestY));

                    const scalar_t frame_here = frame_a[n][c][y][x];

                    if ((northwestX >= 0) && (northwestX < X) && (northwestY >= 0) && (northwestY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][northwestY][northwestX];
                        grad_frame_a[n][c][y][x] += northwest * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((float) (-1.0)) * ((float) (southeastY) - outputY   );
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * ((float) (southeastX) - outputX   ) * ((float) (-1.0));
                    }
                    if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) && (northeastY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][northeastY][northeastX];
                        grad_frame_a[n][c][y][x] += northeast * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((float) (+1.0)) * ((float) (southwestY) - outputY   );
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * (outputX    - (float) (southwestX)) * ((float) (-1.0));
                    }
                    if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) && (southwestY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][southwestY][southwestX];
                        grad_frame_a[n][c][y][x] += southwest * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((float) (-1.0)) * (outputY    - (float) (northeastY));
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * ((float) (northeastX) - outputX   ) * ((float) (+1.0));
                    }
                    if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) && (southeastY < Y)) {
                        const scalar_t grad_output_here = grad_output_a[n][c][southeastY][southeastX];
                        grad_frame_a[n][c][y][x] += southeast * grad_output_here;
                        grad_flow_a[n][0][y][x] += frame_here * grad_output_here * ((float) (+1.0)) * (outputY    - (float) (northwestY));
                        grad_flow_a[n][1][y][x] += frame_here * grad_output_here * (outputX    - (float) (northwestX)) * ((float) (+1.0));
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
            return splatting_forward_cpu_impl<scalar_t>(frame, flow, output);
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
            return splatting_backward_cpu_impl<scalar_t>(frame, flow, grad_output, grad_frame, grad_flow);
        }
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splatting_forward", &splatting_forward, "splatting forward");
    m.def("splatting_backward", &splatting_backward, "splatting backward");
}
