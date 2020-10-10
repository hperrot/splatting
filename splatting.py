import torch
from typing import Union

try:
    import splatting_cpp
except ImportError:
    # try JIT-compilation with ninja
    from torch.utils.cpp_extension import load
    splatting_cpp = load(
        name='splatting_cpp',
        sources=['cpp/splatting.cpp'],
        verbose=True,
    )


class SummationSplattingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, frame, flow):
        assert(frame.dtype == flow.dtype)
        assert(len(frame.shape) == 4)
        assert(len(flow.shape) == 4)
        assert(frame.shape[0] == flow.shape[0])
        assert(frame.shape[2] == flow.shape[2])
        assert(frame.shape[3] == flow.shape[3])
        assert(flow.shape[1] == 2)
        ctx.save_for_backward(frame, flow)
        output = torch.zeros_like(frame)
        splatting_cpp.splatting_forward(frame, flow, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        frame, flow = ctx.saved_tensors
        grad_frame = torch.zeros_like(frame)
        grad_flow = torch.zeros_like(flow)
        splatting_cpp.splatting_backward(frame, flow, grad_output, grad_frame, grad_flow)
        return grad_frame, grad_flow


SPLATTING_TYPES = ["summation", "average", "linear", "softmax"]


def splatting_function(
    splatting_type : str,
    frame : torch.Tensor,
    flow : torch.Tensor,
    importance_metric :  Union[torch.Tensor, None] = None,
    eps : float = 1e-7,
) -> torch.Tensor:
    assert(splatting_type in SPLATTING_TYPES)
    if splatting_type == "summation":
        assert(importance_metric is None)
    elif splatting_type == "average":
        assert(importance_metric is None)
        importance_metric = frame.new_ones([frame.shape[0], 1, frame.shape[2], frame.shape[3]])
        frame = torch.cat([frame, importance_metric], 1)
    elif splatting_type == "linear":
        assert(isinstance(importance_metric, torch.Tensor))
        assert(importance_metric.shape[0] == frame.shape[0])
        assert(importance_metric.shape[1] == 1)
        assert(importance_metric.shape[2] == frame.shape[2])
        assert(importance_metric.shape[3] == frame.shape[3])
        frame = torch.cat([frame * importance_metric, importance_metric], 1)
    elif splatting_type == "softmax":
        assert(isinstance(importance_metric, torch.Tensor))
        assert(importance_metric.shape[0] == frame.shape[0])
        assert(importance_metric.shape[1] == 1)
        assert(importance_metric.shape[2] == frame.shape[2])
        assert(importance_metric.shape[3] == frame.shape[3])
        importance_metric = importance_metric.exp()
        frame = torch.cat([frame * importance_metric, importance_metric], 1)

    output = SummationSplattingFunction.apply(frame, flow)

    if splatting_type != "summation":
        output = output[:, :-1, :, :] / (output[:, -1:, :, :] + eps)

    return output


class Splatting(torch.nn.Module):
    def __init__(self, splatting_type : str, eps : float=1e-7):
        super().__init__()
        assert(splatting_type in SPLATTING_TYPES)
        self.splatting_type = splatting_type
        self.eps = eps

    def forward(
        self,
        frame : torch.Tensor,
        flow : torch.Tensor,
        importance_metric :  Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        return splatting_function(
            self.splatting_type, frame, flow, importance_metric, self.eps
        )
