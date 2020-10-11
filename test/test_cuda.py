import sys

sys.path.append(".")
import torch
import splatting
import splatting.cuda as splatting_cuda
import pytest


def dispatch_fail(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype).cuda()
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype).cuda()
    output = torch.zeros_like(frame)
    with pytest.raises(RuntimeError):
        splatting_cuda.splatting_forward_cuda(frame, flow, output)


def dispatch_not_fail(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype).cuda()
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype).cuda()
    output = torch.zeros_like(frame)
    splatting_cuda.splatting_forward_cuda(frame, flow, output)


class TestForward:
    def test_zero_flow(self):
        frame = torch.ones(1, 3, 2, 2).cuda()
        flow = torch.zeros(1, 2, 2, 2).cuda()
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, frame)

    def test_flow_one(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 1] = 1
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_flow_two(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 2
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 2, 2] = 1
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_two_values_one_target_location(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        frame[0, :, 1, 1] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 1] = 2
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_direction_2(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 1, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 0] = 1
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_direction_3(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 0, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 0, 1] = 1
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_center(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 0.5
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, :2, :2] = 0.25
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_partial(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 1, 0, 0] = 0.2
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 0, 0] = 0.8
        target[0, :, 1, 0] = 0.2
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_out_of_bounds(self):
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 10
        target = torch.zeros(1, 1, 3, 3).cuda()
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_dispatch_fail_and_not_fail(self):
        dispatch_fail(torch.int)
        dispatch_fail(torch.int8)
        dispatch_fail(torch.int16)
        dispatch_fail(torch.int32)
        dispatch_fail(torch.int64)
        dispatch_not_fail(torch.float16)
        dispatch_not_fail(torch.float32)
        dispatch_not_fail(torch.float64)

    def test_not_quadratic(self):
        frame = torch.ones(1, 3, 4, 5).cuda()
        flow = torch.zeros(1, 2, 4, 5).cuda()
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, frame)

    def test_direction_2_not_contiguous(self):
        frame = torch.zeros(1, 1, 3, 3).permute(0, 1, 3, 2).cuda()
        assert not frame.is_contiguous()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 1, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 0] = 1
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)

    def test_direction_3_not_contiguous(self):
        frame = torch.zeros(1, 1, 3, 3).permute(0, 1, 3, 2).cuda()
        assert not frame.is_contiguous()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 0, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 0, 1] = 1
        output = torch.zeros_like(frame)
        splatting_cuda.splatting_forward_cuda(frame, flow, output)
        assert torch.equal(output, target)


class TestBackward:
    def test_grads(self):
        # zero flow
        frame = torch.ones(1, 1, 3, 3).cuda()
        flow = torch.zeros(1, 2, 3, 3).cuda()
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        output.sum().backward()
        grad_frame_target = torch.ones(1, 1, 3, 3).cuda()
        grad_flow_target = torch.zeros(1, 2, 3, 3).cuda()
        grad_flow_target[:, 0, :, -1] = -1
        grad_flow_target[:, 1, -1, :] = -1
        assert torch.allclose(frame.grad, grad_frame_target)
        assert torch.allclose(flow.grad, grad_flow_target)

        # flow ones 0
        frame = torch.ones(1, 1, 3, 3).cuda()
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 0, :, :] = 1
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        output.sum().backward()
        grad_frame_target = torch.ones(1, 1, 3, 3).cuda()
        grad_frame_target[:, :, :, -1] = 0
        grad_flow_target = torch.zeros(1, 2, 3, 3).cuda()
        grad_flow_target[:, 0, :, -2] = -1
        grad_flow_target[:, 1, -1, :-1] = -1
        assert torch.allclose(frame.grad, grad_frame_target)
        assert torch.allclose(flow.grad, grad_flow_target)

        # flow ones 1
        frame = torch.ones(1, 1, 3, 3).cuda()
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 1, :, :] = 1
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        output.sum().backward()
        grad_frame_target = torch.ones(1, 1, 3, 3).cuda()
        grad_frame_target[:, :, -1, :] = 0
        grad_flow_target = torch.zeros(1, 2, 3, 3).cuda()
        grad_flow_target[:, 0, :-1, -1] = -1
        grad_flow_target[:, 1, -2, :] = -1
        assert torch.allclose(frame.grad, grad_frame_target)
        assert torch.allclose(flow.grad, grad_flow_target)
