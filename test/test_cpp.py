import torch
import cpp.splatting_cpp
import pytest


def test_zero_flow():
    frame = torch.ones(1, 3, 2, 2)
    flow = torch.zeros(1, 2, 2, 2)
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, frame))

def test_flow_one():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 1, 1] = 1
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, target))

def test_two_values_one_target_location():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    frame[0, :, 1, 1] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 1, 1] = 2
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, target))

def test_direction_2():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 1, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 1, 0] = 1
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, target))

def test_direction_3():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 0, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 0, 1] = 1
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, target))

def test_center():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 0.5
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, :2, :2] = 0.25
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, target))

def test_partial():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 1, 0, 0] = 0.2
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 0, 0] = 0.8
    target[0, :, 1, 0] = 0.2
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, target))

def test_out_of_bounds():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 10
    target = torch.zeros(1, 1, 3, 3)
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)
    assert(torch.equal(output, target))

def dispatch_fail(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype)
    flow = torch.zeros(1, 1, 3, 3, dtype=dtype)
    with pytest.raises(RuntimeError):
        output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)

def dispatch_not_fail(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype)
    flow = torch.zeros(1, 1, 3, 3, dtype=dtype)
    output = cpp.splatting_cpp.SplattingFunction.apply(frame, flow)

def test_dispatch_fail_and_not_fail():
    dispatch_fail(torch.int)
    dispatch_fail(torch.int8)
    dispatch_fail(torch.int16)
    dispatch_fail(torch.int32)
    dispatch_fail(torch.int64)
    dispatch_fail(torch.float16)
    dispatch_not_fail(torch.float32)
    dispatch_not_fail(torch.float64)
