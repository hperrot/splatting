import torch
import cpp.splatting_cpp
import timeit


def run_test(batch_size, spatial_size, flow_init, repetitions=10):
    frame = torch.ones(batch_size, 3, spatial_size, spatial_size)
    if flow_init == "zeros":
        flow = torch.zeros(batch_size, 2, spatial_size, spatial_size)
    elif flow_init == "ones":
        flow = torch.ones(batch_size, 2, spatial_size, spatial_size)
    else:
        raise Exception
    ex_time = timeit.timeit(
        lambda: cpp.splatting_cpp.SplattingFunction.apply(frame, flow),
        number=repetitions,
    ) / repetitions
    print(f"{batch_size=}\t{spatial_size=}\t{flow_init=}\t{ex_time=}")


def benchmark():
    for batch_size in [1, 2, 4]:
        for spatial_size in [2, 8, 16, 256, 1024]:
            for flow_init in ["zeros", "ones"]:
                run_test(batch_size, spatial_size, flow_init)


if __name__ == "__main__":
    benchmark()
