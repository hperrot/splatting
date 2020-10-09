import torch
import timeit
import sys
sys.path.append(".")

def run_test_forward(method, batch_size, spatial_size, flow_init, repetitions=10):
    frame = torch.ones(batch_size, 3, spatial_size, spatial_size)
    if flow_init == "zeros":
        flow = torch.zeros(batch_size, 2, spatial_size, spatial_size)
    elif flow_init == "ones":
        flow = torch.ones(batch_size, 2, spatial_size, spatial_size)
    else:
        raise NotImplementedError
    if method == "splatting_cpp":
        import splatting_cpp
        output = torch.zeros_like(frame)
        def test_fn():
            splatting_cpp.splatting_forward(frame, flow, output)
    elif method == "splatting_function":
        import splatting
        def test_fn():
            splatting.SplattingFunction.apply(frame, flow)
    elif method == "splatting_module":
        import splatting
        splatting_module = splatting.Splatting()
        def test_fn():
            splatting_module(frame, flow)
    else:
        raise NotImplementedError
    ex_time = timeit.timeit(
        test_fn,
        number=repetitions,
    ) / repetitions
    print(f"forward \t{batch_size=}\t{spatial_size=}\t{flow_init=}\t{ex_time=}")


def run_test_backward(method, batch_size, spatial_size, flow_init, repetitions=10):
    frame = torch.ones(batch_size, 3, spatial_size, spatial_size)
    if flow_init == "zeros":
        flow = torch.zeros(batch_size, 2, spatial_size, spatial_size)
    elif flow_init == "ones":
        flow = torch.ones(batch_size, 2, spatial_size, spatial_size)
    else:
        raise NotImplementedError
    if method == "splatting_cpp":
        import splatting_cpp
        grad_output = torch.zeros_like(frame)
        grad_frame = torch.zeros_like(frame)
        grad_flow = torch.zeros_like(flow)
        def test_fn():
            splatting_cpp.splatting_backward(frame, flow, grad_output, grad_frame, grad_flow)
    elif method == "splatting_function":
        import splatting
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        output = splatting.SplattingFunction.apply(frame, flow).sum()
        def test_fn():
            output.backward(retain_graph=True)
    elif method == "splatting_module":
        import splatting
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        splatting_module = splatting.Splatting()
        output = splatting_module(frame, flow).sum()
        def test_fn():
            output.backward(retain_graph=True)
    else:
        raise NotImplementedError
    ex_time = timeit.timeit(
        test_fn,
        number=repetitions,
    ) / repetitions
    print(f"backward\t{batch_size=}\t{spatial_size=}\t{flow_init=}\t{ex_time=}")


def benchmark(method):
    for batch_size in [1, 2, 4]:
        for spatial_size in [2, 8, 16, 256, 1024]:
            for flow_init in ["zeros", "ones"]:
                run_test_forward(method, batch_size, spatial_size, flow_init)
                run_test_backward(method, batch_size, spatial_size, flow_init)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark splatting implementation")
    parser.add_argument('method', type=str, choices=['splatting_cpp', 'splatting_function', 'splatting_module'], help="What to benchmark")

    args = parser.parse_args()

    benchmark(args.method)
