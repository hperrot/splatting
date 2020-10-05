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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splatting_forward", &splatting_forward, "splatting forward");
}
