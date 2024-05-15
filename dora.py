# Code from: https://github.com/catid/dora 

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# This layer is dropped into pre-trained PyTorch model where nn.Linear is used
class DoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        # m = Magnitude column-wise across output dimension
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
        
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, rank)*std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        lora = torch.matmul(self.lora_A, self.lora_B)
        adapted = self.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)
    
def replace_linear_with_dora(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Get the input and output dimensions of the current nn.Linear layer
            d_in = module.in_features
            d_out = module.out_features

            # Create a new DoRALayer with the same dimensions
            setattr(model, name, DoRALayer(d_out=d_out, d_in=d_in, weight=module.weight.data.clone(), bias=module.bias.data.clone()))
        else:
            # Recursively apply this function to submodules
            replace_linear_with_dora(module)

