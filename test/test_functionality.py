import sys

sys.path.append(".")
import torch
import pytest
import splatting


TEST_CUDA = torch.cuda.is_available() & (splatting.splatting.splatting_cuda is not None)


def dispatch_fail(dtype, device):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype, device=device)
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype, device=device)
    with pytest.raises(RuntimeError):
        splatting.SummationSplattingFunction.apply(frame, flow)


def dispatch_not_fail(dtype, device):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype, device=device)
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype, device=device)
    splatting.SummationSplattingFunction.apply(frame, flow)


class TestCpu:
    device = "cpu"

    def test_zero_flow(self):
        frame = torch.ones(1, 3, 2, 2, device=self.device)
        flow = torch.zeros(1, 2, 2, 2, device=self.device)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, frame)

    def test_flow_one(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 1, 1] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_flow_two(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, :, 0, 0] = 2
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 2, 2] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_two_values_one_target_location(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        frame[0, :, 1, 1] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 1, 1] = 2
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_direction_2(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, 1, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 1, 0] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_direction_3(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, 0, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 0, 1] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_center(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, :, 0, 0] = 0.5
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, :2, :2] = 0.25
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_partial(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, 1, 0, 0] = 0.2
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 0, 0] = 0.8
        target[0, :, 1, 0] = 0.2
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_out_of_bounds(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, :, 0, 0] = 10
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_dispatch_fail_and_not_fail(self):
        dispatch_fail(torch.int, self.device)
        dispatch_fail(torch.int8, self.device)
        dispatch_fail(torch.int16, self.device)
        dispatch_fail(torch.int32, self.device)
        dispatch_fail(torch.int64, self.device)
        dispatch_not_fail(torch.float16, self.device)
        dispatch_not_fail(torch.float32, self.device)
        dispatch_not_fail(torch.float64, self.device)

    def test_not_quadratic(self):
        frame = torch.ones(1, 3, 4, 5, device=self.device)
        flow = torch.zeros(1, 2, 4, 5, device=self.device)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, frame)

    def test_direction_2_not_contiguous(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device).permute(0, 1, 3, 2)
        assert not frame.is_contiguous()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, 1, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 1, 0] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_direction_3_not_contiguous(self):
        frame = torch.zeros(1, 1, 3, 3, device=self.device).permute(0, 1, 3, 2)
        assert not frame.is_contiguous()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, 0, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3, device=self.device)
        target[0, :, 0, 1] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_grads(self):
        # zero flow
        frame = torch.ones(1, 1, 3, 3, device=self.device)
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        output.sum().backward()
        grad_frame_target = torch.ones(1, 1, 3, 3, device=self.device)
        grad_flow_target = torch.zeros(1, 2, 3, 3, device=self.device)
        grad_flow_target[:, 0, :, -1] = -1
        grad_flow_target[:, 1, -1, :] = -1
        assert torch.allclose(frame.grad, grad_frame_target)
        assert torch.allclose(flow.grad, grad_flow_target)

        # flow ones 0
        frame = torch.ones(1, 1, 3, 3, device=self.device)
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, 0, :, :] = 1
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        output.sum().backward()
        grad_frame_target = torch.ones(1, 1, 3, 3, device=self.device)
        grad_frame_target[:, :, :, -1] = 0
        grad_flow_target = torch.zeros(1, 2, 3, 3, device=self.device)
        grad_flow_target[:, 0, :, -2] = -1
        grad_flow_target[:, 1, -1, :-1] = -1
        assert torch.allclose(frame.grad, grad_frame_target)
        assert torch.allclose(flow.grad, grad_flow_target)

        # flow ones 1
        frame = torch.ones(1, 1, 3, 3, device=self.device)
        flow = torch.zeros(1, 2, 3, 3, device=self.device)
        flow[0, 1, :, :] = 1
        frame.requires_grad_(True)
        flow.requires_grad_(True)
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        output.sum().backward()
        grad_frame_target = torch.ones(1, 1, 3, 3, device=self.device)
        grad_frame_target[:, :, -1, :] = 0
        grad_flow_target = torch.zeros(1, 2, 3, 3, device=self.device)
        grad_flow_target[:, 0, :-1, -1] = -1
        grad_flow_target[:, 1, -2, :] = -1
        assert torch.allclose(frame.grad, grad_frame_target)
        assert torch.allclose(flow.grad, grad_flow_target)


@pytest.mark.skipif(not TEST_CUDA, reason="CUDA not available")
class TestCuda(TestCpu):
    device = "cuda"
