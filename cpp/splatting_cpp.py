import torch

import splatting_cpp


class SplattingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, frame, flow):
        ctx.save_for_backward(frame, flow)
        output = splatting_cpp.splatting_forward(frame, flow)
        return output


class Splatting(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frame, flow):
        return SplattingFunction.apply(frame, flow)
