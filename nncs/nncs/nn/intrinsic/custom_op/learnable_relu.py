from functools import partial
import torch
import torch.nn as nn


class _LearnableReLU_torch180(nn.Module):
    def __init__(self, init_thres, inplace=False):
        super(_LearnableReLU_torch180, self).__init__()
        self.register_buffer(
            "clip_val", torch.tensor([init_thres], dtype=torch.float32)
        )
        self.register_buffer("lower_val", torch.tensor([0], dtype=torch.float32))
        self.inplace = inplace

    def forward(self, x):
        output = torch.clamp(x, self.lower_val.data.item(), self.clip_val.data.item())
        return output

    def __repr__(self):
        return "{0}(clip_val={1}, inplace={2})".format(
            self.__class__.__name__, self.clip_val.item(), self.inplace
        )


LearnableReLU6_torch180 = partial(_LearnableReLU_torch180, init_thres=6)
LearnableReLU_torch180 = partial(_LearnableReLU_torch180, init_thres=8)


class _LearnableReLU(nn.Module):
    def __init__(self, init_thres, inplace=False):
        super(_LearnableReLU, self).__init__()
        self.clip_val = nn.Parameter(torch.Tensor([init_thres]))
        self.register_buffer("lower_val", torch.tensor([0], dtype=torch.float32))
        self.inplace = inplace

    def forward(self, x):
        output = torch.clamp(x, self.lower_val, self.clip_val)

        return output

    def __repr__(self):
        return "{0}(clip_val={1}, inplace={2})".format(
            self.__class__.__name__, self.clip_val.item(), self.inplace
        )


LearnableReLU6 = partial(_LearnableReLU, init_thres=6)
LearnableReLU = partial(_LearnableReLU, init_thres=8)


class _LearnableClip_torch180(nn.Module):
    def __init__(self, init_lower, init_upper, inplace=False):
        super(_LearnableClip_torch180, self).__init__()
        self.register_buffer(
            "lower_val", torch.tensor([init_lower], dtype=torch.float32)
        )
        self.register_buffer(
            "upper_val", torch.tensor([init_upper], dtype=torch.float32)
        )
        self.inplace = inplace

    def forward(self, x):
        out = torch.clamp(x, self.lower_val.data.item(), self.upper_val.data.item())
        return out

    def __repr__(self):
        return "{0}(lower_val={1}, upper_val={2}, inplace={3})".format(
            self.__class__.__name__,
            self.lower_val.item(),
            self.upper_val.item(),
            self.inplace,
        )


LearnableClip8_torch180 = partial(_LearnableClip_torch180, init_lower=-8, init_upper=8)


class _LearnableClip(nn.Module):
    def __init__(self, init_lower, init_upper):
        super(_LearnableClip, self).__init__()
        self.lower_val = nn.Parameter(torch.Tensor([init_lower]))
        self.upper_val = nn.Parameter(torch.Tensor([init_upper]))

    def forward(self, x):
        out = torch.clamp(x, self.lower_val, self.upper_val)

        return out

    def __repr__(self):
        return "{0}(lower_val={1}, upper_val={2})".format(
            self.__class__.__name__, self.lower_val.item(), self.upper_val.item()
        )


LearnableClip8 = partial(_LearnableClip, init_lower=-8, init_upper=8)
