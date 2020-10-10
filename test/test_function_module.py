import torch
import splatting
import pytest


class TestSummationSplattingFunction:
    def test_wrong_dtype_combination(self):
        # correct dtype
        frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
        output = splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.dtype != flow.dtype
        frame = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
        with pytest.raises(AssertionError):
            output = splatting.SummationSplattingFunction.apply(frame, flow)

    def test_wrong_dimensions(self):
        # correct shape
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 3, 3)
        output = splatting.SummationSplattingFunction.apply(frame, flow)

        # len(frame.size()) != 4
        frame = torch.zeros(1)
        flow = torch.zeros(1, 2, 3, 3)
        with pytest.raises(AssertionError):
            output = splatting.SummationSplattingFunction.apply(frame, flow)

        # len(flow.size()) != 4
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1)
        with pytest.raises(AssertionError):
            output = splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.size()[0] != flow.size()[0]
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(2, 2, 3, 3)
        with pytest.raises(AssertionError):
            output = splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.size()[2] != flow.size()[2]
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 2, 3)
        with pytest.raises(AssertionError):
            output = splatting.SummationSplattingFunction.apply(frame, flow)

        # frame.size()[3] != flow.size()[3]
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 3, 2)
        with pytest.raises(AssertionError):
            output = splatting.SummationSplattingFunction.apply(frame, flow)

        # flow.size()[1] != 2
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 1, 3, 3)
        with pytest.raises(AssertionError):
            output = splatting.SummationSplattingFunction.apply(frame, flow)


class Test_splatting_function():
    def test_splatting_types(self):
        frame = torch.zeros(1, 1, 3, 3)
        flow = torch.zeros(1, 2, 3, 3)
        importance_metric = torch.ones_like(frame)
        splatting.splatting_function("summation", frame, flow)
        splatting.splatting_function("average", frame, flow)
        splatting.splatting_function("linear", frame, flow, importance_metric)
        splatting.splatting_function("softmax", frame, flow, importance_metric)
        with pytest.raises(NotImplementedError):
            splatting.splatting_function("something_else", frame, flow, importance_metric)


class TestSplatting():
    def test_splatting_types(self):
        splatting.Splatting("summation")
        splatting.Splatting("average")
        splatting.Splatting("linear")
        splatting.Splatting("softmax")
        with pytest.raises(NotImplementedError):
            splatting.Splatting("something_else")
