import torch
import splatting
import splatting_cpp
import pytest


def test_zero_flow():
    frame = torch.ones(1, 3, 2, 2)
    flow = torch.zeros(1, 2, 2, 2)
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, frame))

def test_flow_one():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 1, 1] = 1
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_two_values_one_target_location():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    frame[0, :, 1, 1] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 1, 1] = 2
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_direction_2():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 1, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 1, 0] = 1
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_direction_3():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 0, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 0, 1] = 1
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_center():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 0.5
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, :2, :2] = 0.25
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_partial():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 1, 0, 0] = 0.2
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 0, 0] = 0.8
    target[0, :, 1, 0] = 0.2
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_out_of_bounds():
    frame = torch.zeros(1, 1, 3, 3)
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 10
    target = torch.zeros(1, 1, 3, 3)
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def dispatch_fail(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype)
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype)
    with pytest.raises(RuntimeError):
        output = splatting.SplattingFunction.apply(frame, flow)

def dispatch_not_fail(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype)
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype)
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)

def test_dispatch_fail_and_not_fail():
    dispatch_fail(torch.int)
    dispatch_fail(torch.int8)
    dispatch_fail(torch.int16)
    dispatch_fail(torch.int32)
    dispatch_fail(torch.int64)
    dispatch_not_fail(torch.float16)
    dispatch_not_fail(torch.float32)
    dispatch_not_fail(torch.float64)

def test_not_quadratic():
    frame = torch.ones(1, 3, 4, 5)
    flow = torch.zeros(1, 2, 4, 5)
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, frame))

def test_direction_2_not_contiguous():
    frame = torch.zeros(1, 1, 3, 3).permute(0, 1, 3, 2)
    assert(not frame.is_contiguous())
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 1, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 1, 0] = 1
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_direction_3_not_contiguous():
    frame = torch.zeros(1, 1, 3, 3).permute(0, 1, 3, 2)
    assert(not frame.is_contiguous())
    frame[0, :, 0, 0] = 1
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 0, 0, 0] = 1
    target = torch.zeros(1, 1, 3, 3)
    target[0, :, 0, 1] = 1
    output = torch.zeros_like(frame)
    splatting_cpp.splatting_forward(frame, flow, output)
    assert(torch.equal(output, target))

def test_wrong_dimensions():
    # frame.dtype != flow.dtype
    frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    flow = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    with pytest.raises(AssertionError):
        output = splatting.SplattingFunction.apply(frame, flow)

    # len(frame.size()) != 4
    frame = torch.zeros(1)
    flow = torch.zeros(1, 2, 3, 3)
    with pytest.raises(AssertionError):
        output = splatting.SplattingFunction.apply(frame, flow)

    # len(flow.size()) != 4
    frame = torch.zeros(1, 1, 3, 3)
    flow = torch.zeros(1)
    with pytest.raises(AssertionError):
        output = splatting.SplattingFunction.apply(frame, flow)

    # frame.size()[0] != flow.size()[0]
    frame = torch.zeros(1, 1, 3, 3)
    flow = torch.zeros(2, 2, 3, 3)
    with pytest.raises(AssertionError):
        output = splatting.SplattingFunction.apply(frame, flow)

    # frame.size()[2] != flow.size()[2]
    frame = torch.zeros(1, 1, 3, 3)
    flow = torch.zeros(1, 2, 2, 3)
    with pytest.raises(AssertionError):
        output = splatting.SplattingFunction.apply(frame, flow)

    # frame.size()[3] != flow.size()[3]
    frame = torch.zeros(1, 1, 3, 3)
    flow = torch.zeros(1, 2, 3, 2)
    with pytest.raises(AssertionError):
        output = splatting.SplattingFunction.apply(frame, flow)

    # flow.size()[1] != 2
    frame = torch.zeros(1, 1, 3, 3)
    flow = torch.zeros(1, 1, 3, 3)
    with pytest.raises(AssertionError):
        output = splatting.SplattingFunction.apply(frame, flow)

def test_backward_produces_grads():
    frame = torch.ones(1, 1, 3, 3, requires_grad=True)
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, :, 0, 0] = 0.1
    flow.requires_grad_(True)
    output = splatting.SplattingFunction.apply(frame, flow)
    output.sum().backward()

def test_grads():
    # zero flow
    frame = torch.ones(1, 1, 3, 3, requires_grad=True)
    flow = torch.zeros(1, 2, 3, 3)
    flow.requires_grad_(True)
    output = splatting.SplattingFunction.apply(frame, flow)
    output.sum().backward()
    grad_frame_target = torch.ones(1, 1, 3, 3)
    grad_flow_target = torch.zeros(1, 2, 3, 3)
    grad_flow_target[:, 0, :, -1] = -1
    grad_flow_target[:, 1, -1, :] = -1
    assert(torch.allclose(frame.grad, grad_frame_target))
    assert(torch.allclose(flow.grad, grad_flow_target))

    # flow ones 0
    frame = torch.ones(1, 1, 3, 3, requires_grad=True)
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 0, :, :] = 1
    flow.requires_grad_(True)
    output = splatting.SplattingFunction.apply(frame, flow)
    output.sum().backward()
    grad_frame_target = torch.ones(1, 1, 3, 3)
    grad_frame_target[:, :, :, -1] = 0
    grad_flow_target = torch.zeros(1, 2, 3, 3)
    grad_flow_target[:, 0, :, -2] = -1
    grad_flow_target[:, 1, -1, :-1] = -1
    assert(torch.allclose(frame.grad, grad_frame_target))
    assert(torch.allclose(flow.grad, grad_flow_target))

    # flow ones 1
    frame = torch.ones(1, 1, 3, 3, requires_grad=True)
    flow = torch.zeros(1, 2, 3, 3)
    flow[0, 1, :, :] = 1
    flow.requires_grad_(True)
    output = splatting.SplattingFunction.apply(frame, flow)
    output.sum().backward()
    grad_frame_target = torch.ones(1, 1, 3, 3)
    grad_frame_target[:, :, -1, :] = 0
    grad_flow_target = torch.zeros(1, 2, 3, 3)
    grad_flow_target[:, 0, :-1, -1] = -1
    grad_flow_target[:, 1, -2, :] = -1
    assert(torch.allclose(frame.grad, grad_frame_target))
    assert(torch.allclose(flow.grad, grad_flow_target))
