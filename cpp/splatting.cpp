#include <torch/extension.h>

#include <vector>

#include <cmath>


template<typename scalar_t>
torch::Tensor splatting_forward_cpu_impl(
    const torch::Tensor frame,
    const torch::Tensor flow
) {
    // init output
    torch::Tensor output = torch::zeros_like(frame);
    // create accessors
    auto frame_a = frame.accessor<scalar_t, 4>();
    auto flow_a = flow.accessor<scalar_t, 4>();
    auto output_a = output.accessor<scalar_t, 4>();
    // actual splatting
    for (auto N=0; N<frame.size(0); N++){
        for (auto C=0; C<frame.size(1); C++) {
            for(auto Y=0; Y<frame.size(2); Y++) {
                for(auto X=0; X<frame.size(3); X++) {

                    const float outputX = (float) X + flow_a[N][0][Y][X];
                    const float outputY = (float) Y + flow_a[N][1][Y][X];

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

                    if ((northwestX >= 0) & (northwestX < output.size(3)) & (northwestY >= 0) & (northwestY < output.size(2))) {
                        output_a[N][C][northwestY][northwestX] += northwest * frame_a[N][C][Y][X];
                    }
                    if ((northeastX >= 0) & (northeastX < output.size(3)) & (northeastY >= 0) & (northeastY < output.size(2))) {
                        output_a[N][C][northeastY][northeastX] += northeast * frame_a[N][C][Y][X];
                    }
                    if ((southwestX >= 0) & (southwestX < output.size(3)) & (southwestY >= 0) & (southwestY < output.size(2))) {
                        output_a[N][C][southwestY][southwestX] += southwest * frame_a[N][C][Y][X];
                    }
                    if ((southeastX >= 0) & (southeastX < output.size(3)) & (southeastY >= 0) & (southeastY < output.size(2))) {
                        output_a[N][C][southeastY][southeastX] += southeast * frame_a[N][C][Y][X];
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
