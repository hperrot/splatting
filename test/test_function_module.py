import sys
sys.path.append(".")
import torch
import splatting
import pytest
from numpy import exp


def dispatch_fail_cuda(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype).cuda()
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype).cuda()
    with pytest.raises(RuntimeError):
        splatting.SummationSplattingFunction.apply(frame, flow)


def dispatch_not_fail_cuda(dtype):
    frame = torch.zeros(1, 1, 3, 3, dtype=dtype).cuda()
    flow = torch.zeros(1, 2, 3, 3, dtype=dtype).cuda()
    splatting.SummationSplattingFunction.apply(frame, flow)


class TestSummationSplattingFunction:
    def test_wrong_dtype_combination(self):
        # correct dtype
        frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
        splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.dtype != flow.dtype
        frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

    def test_wrong_device_combination(self):
        if not torch.cuda.is_available():
            pytest.skip()
        # both cpu
        frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
        splatting.SummationSplattingFunction.apply(frame, flow)

        # both gpu
        frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32).cuda()
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float32).cuda()
        splatting.SummationSplattingFunction.apply(frame, flow)

        # frmae.device != flow.device
        frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float32).cuda()
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

    def test_wrong_dimensions(self):
        # correct shape
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 3, 3)
        splatting.SummationSplattingFunction.apply(frame, flow)

        # len(frame.size()) != 4
        frame = torch.zeros(1)
        flow = torch.zeros(1, 2, 3, 3)
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

        # len(flow.size()) != 4
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1)
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.size()[0] != flow.size()[0]
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(2, 2, 3, 3)
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.size()[2] != flow.size()[2]
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 2, 3)
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.size()[3] != flow.size()[3]
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 3, 2)
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

        # flow.size()[1] != 2
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 1, 3, 3)
        with pytest.raises(AssertionError):
            splatting.SummationSplattingFunction.apply(frame, flow)

    def test_flow_one_cpp(self):
        frame = torch.zeros(1, 1, 3, 3)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3)
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3)
        target[0, :, 1, 1] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_flow_one_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 1] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_flow_two_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 2
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 2, 2] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_zero_flow_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.ones(1, 3, 2, 2).cuda()
        flow = torch.zeros(1, 2, 2, 2).cuda()
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, frame)


    def test_two_values_one_target_location_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        frame[0, :, 1, 1] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 1] = 2
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_direction_2_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 1, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 0] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_direction_3_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 0, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 0, 1] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_center_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 0.5
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, :2, :2] = 0.25
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_partial_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 1, 0, 0] = 0.2
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 0, 0] = 0.8
        target[0, :, 1, 0] = 0.2
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_out_of_bounds_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).cuda()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, :, 0, 0] = 10
        target = torch.zeros(1, 1, 3, 3).cuda()
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_dispatch_fail_and_not_fail_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        dispatch_fail_cuda(torch.int)
        dispatch_fail_cuda(torch.int8)
        dispatch_fail_cuda(torch.int16)
        dispatch_fail_cuda(torch.int32)
        dispatch_fail_cuda(torch.int64)
        dispatch_not_fail_cuda(torch.float16)
        dispatch_not_fail_cuda(torch.float32)
        dispatch_not_fail_cuda(torch.float64)

    def test_not_quadratic_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.ones(1, 3, 4, 5).cuda()
        flow = torch.zeros(1, 2, 4, 5).cuda()
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, frame)

    def test_direction_2_not_contiguous(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).permute(0, 1, 3, 2).cuda()
        assert not frame.is_contiguous()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 1, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 1, 0] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_direction_3_not_contiguous(self):
        if not torch.cuda.is_available():
            pytest.skip()
        frame = torch.zeros(1, 1, 3, 3).permute(0, 1, 3, 2).cuda()
        assert not frame.is_contiguous()
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3).cuda()
        flow[0, 0, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3).cuda()
        target[0, :, 0, 1] = 1
        output = splatting.SummationSplattingFunction.apply(frame, flow)
        assert torch.equal(output, target)

    def test_grads_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip()
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


class Test_splatting_function:
    def test_splatting_type_names(self):
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 3, 3)
        importance_metric = torch.ones_like(frame)
        splatting.splatting_function("summation", frame, flow)
        splatting.splatting_function("average", frame, flow)
        splatting.splatting_function("linear", frame, flow, importance_metric)
        splatting.splatting_function("softmax", frame, flow, importance_metric)
        with pytest.raises(NotImplementedError):
            splatting.splatting_function(
                "something_else", frame, flow, importance_metric
            )

    def test_splatting_type_values(self):
        frame = torch.tensor([1, 2], dtype=torch.float32).reshape([1, 1, 1, 2])
        flow = torch.zeros([1, 2, 1, 2], dtype=torch.float32)
        flow[0, 0, 0, 0] = 1
        importance_metric = torch.tensor([1, 2], dtype=torch.float32).reshape(
            [1, 1, 1, 2]
        )

        # summation splatting
        output = splatting.splatting_function("summation", frame, flow)
        assert output[0, 0, 0, 1] == pytest.approx(3)

        # average splatting
        output = splatting.splatting_function("average", frame, flow)
        assert output[0, 0, 0, 1] == pytest.approx(1.5)

        # linear splatting
        output = splatting.splatting_function("linear", frame, flow, importance_metric)
        assert output[0, 0, 0, 1] == pytest.approx(5.0 / 3.0)

        # softmax splatting
        output = splatting.splatting_function("softmax", frame, flow, importance_metric)
        assert output[0, 0, 0, 1] == pytest.approx(
            (exp(1) + 2 * exp(2)) / (exp(1) + exp(2))
        )

    def test_importance_metric_type_and_shape(self):
        frame = torch.ones([1, 1, 3, 3])
        flow = torch.zeros([1, 2, 3, 3])
        importance_metric = frame.new_ones([1, 1, 3, 3])
        wrong_metric_0 = frame.new_ones([2, 1, 3, 3])
        wrong_metric_1 = frame.new_ones([1, 2, 3, 3])
        wrong_metric_2 = frame.new_ones([1, 1, 2, 3])
        wrong_metric_3 = frame.new_ones([1, 1, 3, 2])

        # summation splatting
        splatting.splatting_function("summation", frame, flow)
        with pytest.raises(AssertionError):
            splatting.splatting_function("summation", frame, flow, importance_metric)

        # average splatting
        splatting.splatting_function("average", frame, flow)
        with pytest.raises(AssertionError):
            splatting.splatting_function("average", frame, flow, importance_metric)

        # linear splatting
        splatting.splatting_function("linear", frame, flow, importance_metric)
        with pytest.raises(AssertionError):
            splatting.splatting_function("linear", frame, flow)
        with pytest.raises(AssertionError):
            splatting.splatting_function("linear", frame, flow, wrong_metric_0)
        with pytest.raises(AssertionError):
            splatting.splatting_function("linear", frame, flow, wrong_metric_1)
        with pytest.raises(AssertionError):
            splatting.splatting_function("linear", frame, flow, wrong_metric_2)
        with pytest.raises(AssertionError):
            splatting.splatting_function("linear", frame, flow, wrong_metric_3)

        # softmax splatting
        splatting.splatting_function("softmax", frame, flow, importance_metric)
        with pytest.raises(AssertionError):
            splatting.splatting_function("softmax", frame, flow)
        with pytest.raises(AssertionError):
            splatting.splatting_function("softmax", frame, flow, wrong_metric_0)
        with pytest.raises(AssertionError):
            splatting.splatting_function("softmax", frame, flow, wrong_metric_1)
        with pytest.raises(AssertionError):
            splatting.splatting_function("softmax", frame, flow, wrong_metric_2)
        with pytest.raises(AssertionError):
            splatting.splatting_function("softmax", frame, flow, wrong_metric_3)

    def test_flow_one(self):
        frame = torch.zeros(1, 1, 3, 3)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3)
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3)
        target[0, :, 1, 1] = 1
        output = splatting.splatting_function("summation", frame, flow)
        assert torch.equal(output, target)


class TestSplatting:
    def test_splatting_types(self):
        splatting.Splatting("summation")
        splatting.Splatting("average")
        splatting.Splatting("linear")
        splatting.Splatting("softmax")
        with pytest.raises(NotImplementedError):
            splatting.Splatting("something_else")

    def test_flow_one(self):
        frame = torch.zeros(1, 1, 3, 3)
        frame[0, :, 0, 0] = 1
        flow = torch.zeros(1, 2, 3, 3)
        flow[0, :, 0, 0] = 1
        target = torch.zeros(1, 1, 3, 3)
        target[0, :, 1, 1] = 1
        output = splatting.Splatting("summation")(frame, flow)
        assert torch.equal(output, target)
