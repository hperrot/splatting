#include <torch/extension.h>
#include <vector>
#include <cmath>


template<typename scalar_t>
torch::Tensor splatting_forward_cpu_impl(
    const torch::Tensor frame,
    const torch::Tensor flow
) {
    torch::Tensor output = torch::zeros_like(frame);

    const scalar_t *frame_ptr = frame.data<scalar_t>();
    const scalar_t *flow_ptr = flow.data<scalar_t>();
    scalar_t *output_ptr = output.data<scalar_t>();

    const auto N = frame.size(0);
    const auto C = frame.size(1);
    const auto Y = frame.size(2);
    const auto X = frame.size(3);

    const auto frame_n_stride = frame.stride(0);
    const auto frame_c_stride = frame.stride(1);
    const auto frame_y_stride = frame.stride(2);
    const auto frame_x_stride = frame.stride(3);
    const auto flow_n_stride = flow.stride(0);
    const auto flow_c_stride = flow.stride(1);
    const auto flow_y_stride = flow.stride(2);
    const auto flow_x_stride = flow.stride(3);
    const auto output_n_stride = output.stride(0);
    const auto output_c_stride = output.stride(1);
    const auto output_y_stride = output.stride(2);
    const auto output_x_stride = output.stride(3);

    for (auto n=0; n<N; n++){
        for (auto c=0; c<C; c++) {
            for(auto y=0; y<Y; y++) {
                for(auto x=0; x<X; x++) {

                    const float outputX = (float) x + flow_ptr[
                        n * flow_n_stride + 0 * flow_c_stride + y * flow_y_stride + x * flow_x_stride
                    ];
                    const float outputY = (float) y + flow_ptr[
                        n * flow_n_stride + 1 * flow_c_stride + y * flow_y_stride + x * flow_x_stride
                    ];

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
                        output_ptr[
                            n * output_n_stride + c * output_c_stride + northwestY * output_y_stride + northwestX * output_x_stride
                        ] += northwest * frame_ptr[
                            n * frame_n_stride + c * frame_c_stride + y * frame_y_stride + x * frame_x_stride
                        ];
                    }
                    if ((northeastX >= 0) && (northeastX < X) && (northeastY >= 0) && (northeastY < Y)) {
                        output_ptr[
                            n * output_n_stride + c * output_c_stride + northeastY * output_y_stride + northeastX * output_x_stride
                        ] += northeast * frame_ptr[
                            n * frame_n_stride + c * frame_c_stride + y * frame_y_stride + x * frame_x_stride
                        ];
                    }
                    if ((southwestX >= 0) && (southwestX < X) && (southwestY >= 0) && (southwestY < Y)) {
                        output_ptr[
                            n * output_n_stride + c * output_c_stride + southwestY * output_y_stride + southwestX * output_x_stride
                        ] += southwest * frame_ptr[
                            n * frame_n_stride + c * frame_c_stride + y * frame_y_stride + x * frame_x_stride
                        ];
                    }
                    if ((southeastX >= 0) && (southeastX < X) && (southeastY >= 0) && (southeastY < Y)) {
                        output_ptr[
                            n * output_n_stride + c * output_c_stride + southeastY * output_y_stride + southeastX * output_x_stride
                        ] += southeast * frame_ptr[
                            n * frame_n_stride + c * frame_c_stride + y * frame_y_stride + x * frame_x_stride
                        ];
                    }

                }
            }
        }
    }

    return output;
}

torch::Tensor splatting_forward(
    const torch::Tensor frame,
    const torch::Tensor flow
) {
    return AT_DISPATCH_FLOATING_TYPES(
        frame.scalar_type(),
        "splatting_forward_cpu",
        [&] {
            return splatting_forward_cpu_impl<scalar_t>(frame, flow);
        }
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splatting_forward", &splatting_forward, "splatting forward");
}
