from shutil import Error
import torch
from typing import Union
import os

splatting_dirname = os.path.dirname(os.path.dirname(__file__))

try:
    from splatting import cpu as splatting_cpu
except ImportError:
    # try JIT-compilation with ninja
    from torch.utils.cpp_extension import load

    splatting_cpu = load(
        name="splatting_cpu",
        sources=[os.path.join(splatting_dirname, "cpp/splatting.cpp")],
        verbose=True,
        extra_cflags=["-O3"],
    )

try:
    from splatting import cuda as splatting_cuda
except ImportError:
    # try JIT-compilation with ninja
    from torch.utils.cpp_extension import load

    try:
        import glob
        import os

        splatting_cuda = load(
            name="splatting_cuda",
            sources=[
                os.path.join(splatting_dirname, "cuda/splatting_cuda.cpp"),
                os.path.join(splatting_dirname, "cuda/splatting.cu"),
            ],
            extra_include_paths=[
                os.path.dirname(
                    glob.glob("/usr/local/**/cublas_v2.h", recursive=True)[0]
                )
            ],
            verbose=True,
            extra_cflags=["-O3"],
        )
    except:
        import warnings

        warnings.warn(
            "splatting.cuda could not be imported nor jit compiled", ImportWarning
        )
        splatting_cuda = None


class SummationSplattingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame, flow):
        assert frame.dtype == flow.dtype
        assert frame.device == flow.device
        assert len(frame.shape) == 4
        assert len(flow.shape) == 4
        assert frame.shape[0] == flow.shape[0]
        assert frame.shape[2] == flow.shape[2]
        assert frame.shape[3] == flow.shape[3]
        assert flow.shape[1] == 2
        ctx.save_for_backward(frame, flow)
        output = torch.zeros_like(frame)
        if frame.is_cuda:
            if splatting_cuda is not None:
                splatting_cuda.splatting_forward_cuda(frame, flow, output)
            else:
                raise RuntimeError("splatting.cuda is not available")
        else:
            splatting_cpu.splatting_forward_cpu(frame, flow, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        frame, flow = ctx.saved_tensors
        grad_frame = torch.zeros_like(frame)
        grad_flow = torch.zeros_like(flow)
        if frame.is_cuda:
            if splatting_cuda is not None:
                splatting_cuda.splatting_backward_cuda(
                    frame, flow, grad_output, grad_frame, grad_flow
                )
            else:
                raise RuntimeError("splatting.cuda is not available")
        else:
            splatting_cpu.splatting_backward_cpu(
                frame, flow, grad_output, grad_frame, grad_flow
            )
        return grad_frame, grad_flow


SPLATTING_TYPES = ["summation", "average", "linear", "softmax"]


def splatting_function(
    splatting_type: str,
    frame: torch.Tensor,
    flow: torch.Tensor,
    importance_metric: Union[torch.Tensor, None] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    if splatting_type == "summation":
        assert importance_metric is None
    elif splatting_type == "average":
        assert importance_metric is None
        importance_metric = frame.new_ones(
            [frame.shape[0], 1, frame.shape[2], frame.shape[3]]
        )
        frame = torch.cat([frame, importance_metric], 1)
    elif splatting_type == "linear":
        assert isinstance(importance_metric, torch.Tensor)
        assert importance_metric.shape[0] == frame.shape[0]
        assert importance_metric.shape[1] == 1
        assert importance_metric.shape[2] == frame.shape[2]
        assert importance_metric.shape[3] == frame.shape[3]
        frame = torch.cat([frame * importance_metric, importance_metric], 1)
    elif splatting_type == "softmax":
        assert isinstance(importance_metric, torch.Tensor)
        assert importance_metric.shape[0] == frame.shape[0]
        assert importance_metric.shape[1] == 1
        assert importance_metric.shape[2] == frame.shape[2]
        assert importance_metric.shape[3] == frame.shape[3]
        importance_metric = importance_metric.exp()
        frame = torch.cat([frame * importance_metric, importance_metric], 1)
    else:
        raise NotImplementedError(
            "splatting_type has to be one of {}, not '{}'".format(
                SPLATTING_TYPES, splatting_type
            )
        )

    output = SummationSplattingFunction.apply(frame, flow)

    if splatting_type != "summation":
        output = output[:, :-1, :, :] / (output[:, -1:, :, :] + eps)

    return output


class Splatting(torch.nn.Module):
    def __init__(self, splatting_type: str, eps: float = 1e-7):
        super().__init__()
        if splatting_type not in SPLATTING_TYPES:
            raise NotImplementedError(
                "splatting_type has to be one of {}, not '{}'".format(
                    SPLATTING_TYPES, splatting_type
                )
            )
        self.splatting_type = splatting_type
        self.eps = eps

    def forward(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor,
        importance_metric: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        return splatting_function(
            self.splatting_type, frame, flow, importance_metric, self.eps
        )
