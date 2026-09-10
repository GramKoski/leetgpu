import torch
import torch.nn as nn


# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    # Note that this is wrong in python: output = model.forward(input) 
    # because it simply makes output refer to a new thing but the original
    # output tensor is never touched
    with torch.no_grad():
        output.copy_(model(input))

