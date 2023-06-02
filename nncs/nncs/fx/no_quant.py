import torch.nn as nn


class FXNNCSIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FXNNCSIdentity, self).__init__()

    def forward(self, input):
        return input


class FXNNCSNoQuant(nn.Module):
    def __init__(self, mod):
        super(FXNNCSNoQuant, self).__init__()
        self.nncs_no_quant = mod
        self.fx_nncs_no_quant = FXNNCSIdentity()

    def forward(self, *args, **kwargs):
        out = self.nncs_no_quant(*args, **kwargs)
        return self.fx_nncs_no_quant(out)
