import torch.nn as nn


class NNCSNoQuant(nn.Module):
    def __init__(self, module):
        super(NNCSNoQuant, self).__init__()
        self.nncs_no_quant = module

    def forward(self, *args, **kwargs):
        return self.nncs_no_quant(*args, **kwargs)
