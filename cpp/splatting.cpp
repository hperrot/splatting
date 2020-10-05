#include <torch/extension.h>
#include <vector>
#include <cmath>


template<typename scalar_t>
torch::Tensor splatting_forward_cpu_impl(
    const torch::Tensor frame,
    const torch::Tensor flow
) {
    torch::Tensor output = torch::zeros_like(frame);

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

    return output;
}


template<typename scalar_t>
std::vector<torch::Tensor> splatting_backward_cpu_impl(
    const torch::Tensor frame,
    const torch::Tensor flow,
    const torch::Tensor grad_output
) {
    torch::Tensor grad_frame = torch::zeros_like(frame);
    torch::Tensor grad_flow = torch::zeros_like(flow);

    const auto frame_a = frame.accessor<scalar_t, 4>();
    const auto flow_a = flow.accessor<scalar_t, 4>();
    const auto grad_output_a = grad_output.accessor<scalar_t, 4>();
    auto grad_frame_a = grad_frame.accessor<scalar_t, 4>();
    auto grad_flow_a = grad_flow.accessor<scalar_t, 4>();

    { // grad_frame
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

                        scalar_t grad_frame_here = 0.0;

                        if ((northwestX >= 0) && (northwestX < X) && (northwestY >= 0) && (northwestY < Y)) {
                            grad_frame_here += northwest * grad_output_a[n][c][northwestY][northwestX];
                        }
                        if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) && (northeastY < Y)) {
                            grad_frame_here += northeast * grad_output_a[n][c][northeastY][northeastX];
                        }
                        if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) && (southwestY < Y)) {
                            grad_frame_here += southwest * grad_output_a[n][c][southwestY][southwestX];
                        }
                        if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) && (southeastY < Y)) {
                            grad_frame_here += southeast * grad_output_a[n][c][southeastY][southeastX];
                        }

                        grad_frame_a[n][c][y][x] = grad_frame_here;

                    }
                }
            }
        }

    } // grad_frame

    { // grad_flow

        const auto N = flow.size(0);
        const auto C = flow.size(1);
        const auto Y = flow.size(2);
        const auto X = flow.size(3);
        const auto C_frame = frame.size(1);

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

                        scalar_t northwest = 0.0;
                        scalar_t northeast = 0.0;
                        scalar_t southwest = 0.0;
                        scalar_t southeast = 0.0;
                        if (c == 0) {
                            northwest = ((float) (-1.0)) * ((float) (southeastY) - outputY   );
                            northeast = ((float) (+1.0)) * ((float) (southwestY) - outputY   );
                            southwest = ((float) (-1.0)) * (outputY    - (float) (northeastY));
                            southeast = ((float) (+1.0)) * (outputY    - (float) (northwestY));
                        } else if (c == 1) {
                            northwest = ((float) (southeastX) - outputX   ) * ((float) (-1.0));
                            northeast = (outputX    - (float) (southwestX)) * ((float) (-1.0));
                            southwest = ((float) (northeastX) - outputX   ) * ((float) (+1.0));
                            southeast = (outputX    - (float) (northwestX)) * ((float) (+1.0));
                        }

                        scalar_t grad_flow_here = 0.0;

                        for (auto c_frame = 0; c_frame < C_frame; c_frame++) {
                            scalar_t frame_here = frame_a[n][c_frame][y][x];

                            if ((northwestX >= 0) && (northwestX < X) && (northwestY >= 0) && (northwestY < Y)) {
                                grad_flow_here += northwest * frame_here * grad_output_a[n][c_frame][northwestY][northwestX];
                            }
                            if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) && (northeastY < Y)) {
                                grad_flow_here += northeast * frame_here * grad_output_a[n][c_frame][northeastY][northeastX];
                            }
                            if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) && (southwestY < Y)) {
                                grad_flow_here += southwest * frame_here * grad_output_a[n][c_frame][southwestY][southwestX];
                            }
                            if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) && (southeastY < Y)) {
                                grad_flow_here += southeast * frame_here * grad_output_a[n][c_frame][southeastY][southeastX];
                            }
                        }

                        grad_flow_a[n][c][y][x] = grad_flow_here;

                    }
                }
            }
        }
    }

    return {grad_frame, grad_flow};
}


torch::Tensor splatting_forward(
    const torch::Tensor frame,
    const torch::Tensor flow
) {
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        frame.scalar_type(),
        "splatting_forward_cpu",
        [&] {
            return splatting_forward_cpu_impl<scalar_t>(frame, flow);
        }
    );
}


std::vector<torch::Tensor> splatting_backward(
    const torch::Tensor frame,
    const torch::Tensor flow,
    const torch::Tensor grad_output
) {
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        frame.scalar_type(),
        "splatting_backward_cpu",
        [&] {
            return splatting_backward_cpu_impl<scalar_t>(frame, flow, grad_output);
        }
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splatting_forward", &splatting_forward, "splatting forward");
    m.def("splatting_backward", &splatting_backward, "splatting backward");
}
