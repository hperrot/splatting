import torch
import splatting_cpp
import timeit


def run_test_forward(batch_size, spatial_size, flow_init, repetitions=10):
    frame = torch.ones(batch_size, 3, spatial_size, spatial_size)
    if flow_init == "zeros":
        flow = torch.zeros(batch_size, 2, spatial_size, spatial_size)
    elif flow_init == "ones":
        flow = torch.ones(batch_size, 2, spatial_size, spatial_size)
    else:
        raise Exception
    output = torch.zeros_like(frame)
    ex_time = timeit.timeit(
        lambda: splatting_cpp.splatting_forward(frame, flow, output),
        number=repetitions,
    ) / repetitions
    print(f"forward \t{batch_size=}\t{spatial_size=}\t{flow_init=}\t{ex_time=}")


def run_test_backward(batch_size, spatial_size, flow_init, repetitions=10):
    frame = torch.ones(batch_size, 3, spatial_size, spatial_size)
    if flow_init == "zeros":
        flow = torch.zeros(batch_size, 2, spatial_size, spatial_size)
    elif flow_init == "ones":
        flow = torch.ones(batch_size, 2, spatial_size, spatial_size)
    else:
        raise Exception
    grad_output = torch.zeros_like(frame)
    grad_frame = torch.zeros_like(frame)
    grad_flow = torch.zeros_like(flow)
    ex_time = timeit.timeit(
        lambda: splatting_cpp.splatting_backward(frame, flow, grad_output, grad_frame, grad_flow),
        number=repetitions,
    ) / repetitions
    print(f"backward\t{batch_size=}\t{spatial_size=}\t{flow_init=}\t{ex_time=}")


def benchmark():
    for batch_size in [1, 2, 4]:
        for spatial_size in [2, 8, 16, 256, 1024]:
            for flow_init in ["zeros", "ones"]:
                run_test_forward(batch_size, spatial_size, flow_init)
                run_test_backward(batch_size, spatial_size, flow_init)


if __name__ == "__main__":
    benchmark()
