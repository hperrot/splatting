import torch

import splatting_cpp


class SplattingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, frame, flow):
        assert(frame.dtype == flow.dtype)
        assert(len(frame.size()) == 4)
        assert(len(flow.size()) == 4)
        assert(frame.size()[0] == flow.size()[0])
        assert(frame.size()[2] == flow.size()[2])
        assert(frame.size()[3] == flow.size()[3])
        assert(flow.size()[1] == 2)
        ctx.save_for_backward(frame, flow)
        output = splatting_cpp.splatting_forward(frame, flow)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        frame, flow = ctx.saved_tensors
        grad_frame, grad_flow = splatting_cpp.splatting_backward(frame, flow, grad_output)
        return grad_frame, grad_flow


class Splatting(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frame, flow):
        return SplattingFunction.apply(frame, flow)
