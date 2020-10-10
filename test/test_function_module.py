import torch
import splatting
import pytest
from numpy import exp


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


class TestSplatting:
    def test_splatting_types(self):
        splatting.Splatting("summation")
        splatting.Splatting("average")
        splatting.Splatting("linear")
        splatting.Splatting("softmax")
        with pytest.raises(NotImplementedError):
            splatting.Splatting("something_else")
