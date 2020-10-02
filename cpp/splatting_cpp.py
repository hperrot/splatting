import torch

import splatting_cpp


class Splatting(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frame, flow):
        pass
