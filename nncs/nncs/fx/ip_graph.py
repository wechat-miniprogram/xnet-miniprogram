import copy

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch import Tensor
from torch.fx import GraphModule
import numpy as np

from nncs.utils import _parent_name


class OpStats(nn.Module):
    def __init__(self):
        super(OpStats, self).__init__()
        self.op = None
        self.fast_mode = True
        self.debug = False

        self._inshape = None
        self._outshape = None
        self._flops = 0

    def forward(self, x: Tensor):
        import ipdb

        ipdb.set_trace()
        return x


class IdentityStats(OpStats):
    def __init__(self):
        super(IdentityStats, self).__init__()

    def forward(self, x: Tensor):
        self._flops = 0
        self._inshape = x.shape
        self._outshape = x.shape
        return self.op(x)


class FlattenStats(OpStats):
    def __init__(self):
        super(FlattenStats, self).__init__()

    def forward(self, x: Tensor):
        self._flops = 0
        shapes = x.shape
        self._inshape = shapes
        if self.fast_mode:
            end_dim = self.op.end_dim
            start_dim = self.op.start_dim
            if end_dim < 0:
                end_dim = len(shapes) + end_dim
            dim = shapes[start_dim]
            for i in range(start_dim + 1, end_dim + 1):
                dim *= shapes[i]
            new_shapes = []
            for i in range(0, start_dim):
                new_shapes.append(shapes[i])
            new_shapes.append(dim)
            self._outshape = tuple(new_shapes)
            return torch.zeros(new_shapes, device=x.device)
        else:
            y = self.op(x)
            self._outshape = tuple(y.shape)
            return self.op(x)


class Conv1dStats(OpStats):
    def __init__(self):
        super(Conv1dStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)
        bs = in_shape[0]
        in_C = in_shape[1]
        in_L = in_shape[2]

        out_channels = self.op.out_channels
        kernel_size = self.op.kernel_size
        stride = self.op.stride
        padding = self.op.padding

        if self.debug or self.fast_mode:
            out_L = (in_L + 2 * padding[0] - kernel_size[0]) // stride[0] + 1

            out_shape0 = (x.shape[0], out_channels, out_L)
            self._outshape = out_shape0

        if self.debug or not self.fast_mode:
            y = self.op(x)
            out_shape1 = y.shape
            self._outshape = tuple(out_shape1)

        if self.debug:
            assert tuple(out_shape1) == out_shape0

        groups = self.op.groups
        filters_per_channel = out_channels // groups
        active_elements_count = bs * np.prod(self._outshape[2:])
        conv_per_position_flops = in_C * filters_per_channel * np.prod(kernel_size)

        overall_conv_flops = conv_per_position_flops * active_elements_count

        if self.op.bias is not None:
            bias_flops = out_channels * active_elements_count
        else:
            bias_flops = 0

        overall_flops = overall_conv_flops + bias_flops
        self._flops = overall_flops

        if self.fast_mode:
            return torch.zeros(out_shape0, device=x.device)
        else:
            return y


class Conv2dStats(OpStats):
    def __init__(self):
        super(Conv2dStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)
        bs = in_shape[0]
        in_channels = in_shape[1]
        h_i = in_shape[2]
        w_i = in_shape[3]

        out_channels = self.op.out_channels
        kernel_size = self.op.kernel_size
        stride = self.op.stride
        padding = self.op.padding
        if self.debug or self.fast_mode:
            h_o = (h_i + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            w_o = (w_i + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

            out_shape0 = (x.shape[0], out_channels, h_o, w_o)
            self._outshape = out_shape0

        if self.debug or not self.fast_mode:
            y = self.op(x)
            out_shape1 = y.shape
            self._outshape = tuple(out_shape1)

        if self.debug:
            assert tuple(out_shape1) == out_shape0

        groups = self.op.groups
        filters_per_channel = out_channels // groups
        active_elements_count = bs * np.prod(self._outshape[2:])
        conv_per_position_flops = (
            in_channels * filters_per_channel * np.prod(kernel_size)
        )

        overall_conv_flops = conv_per_position_flops * active_elements_count

        if self.op.bias is not None:
            bias_flops = out_channels * active_elements_count
        else:
            bias_flops = 0

        overall_flops = overall_conv_flops + bias_flops
        self._flops = overall_flops

        if self.fast_mode:
            return torch.zeros(out_shape0, device=x.device)
        else:
            return y


class BatchNorm2dStats(OpStats):
    def __init__(self):
        super(BatchNorm2dStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)

        if self.debug or self.fast_mode:
            out_shape0 = self._inshape
            self._outshape = out_shape0

        if self.debug or not self.fast_mode:
            y = self.op(x)
            out_shape1 = y.shape
            self._outshape = tuple(out_shape1)

        if self.debug:
            assert tuple(out_shape1) == out_shape0

        self._flops = np.prod(self._inshape) * 2

        if self.fast_mode:
            return torch.zeros(out_shape0, device=x.device)
        else:
            return y


class ReLU6Stats(OpStats):
    def __init__(self):
        super(ReLU6Stats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)

        if self.debug or self.fast_mode:
            out_shape0 = self._inshape
            self._outshape = out_shape0

        if self.debug or not self.fast_mode:
            y = self.op(x)
            out_shape1 = y.shape
            self._outshape = tuple(out_shape1)

        if self.debug:
            assert tuple(out_shape1) == out_shape0

        self._flops = np.prod(self._inshape)

        if self.fast_mode:
            return torch.zeros(out_shape0, device=x.device)
        else:
            return y


class DropoutStats(OpStats):
    def __init__(self):
        super(DropoutStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)

        if self.debug or self.fast_mode:
            out_shape0 = self._inshape
            self._outshape = out_shape0

        if self.debug or not self.fast_mode:
            y = self.op(x)
            out_shape1 = y.shape
            self._outshape = tuple(out_shape1)

        if self.debug:
            assert tuple(out_shape1) == out_shape0

        if self.fast_mode:
            return torch.zeros(out_shape0, device=x.device)
        else:
            return y


class LinearStats(OpStats):
    def __init__(self):
        super(LinearStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)

        # in_features = self.op.in_features
        out_features = self.op.out_features
        if self.debug or self.fast_mode:
            out_shape0 = (x.shape[0], out_features)
            self._outshape = out_shape0

        if self.debug or not self.fast_mode:
            y = self.op(x)
            out_shape1 = y.shape
            self._outshape = tuple(out_shape1)

        if self.debug:
            assert tuple(out_shape1) == out_shape0

        output_last_dim = self._outshape[-1]
        bias_flops = output_last_dim if self.op.bias is not None else 0
        self._flops = np.prod(self._inshape) * output_last_dim + bias_flops

        if self.fast_mode:
            return torch.zeros(out_shape0, device=x.device)
        else:
            return y


class Pool2dStats(OpStats):
    def __init__(self):
        super(Pool2dStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)
        bs = in_shape[0]
        in_channels = in_shape[1]
        h_i = in_shape[2]
        w_i = in_shape[3]

        kernel_size = _pair(self.op.kernel_size)
        stride = _pair(self.op.stride)
        padding = _pair(self.op.padding)
        if self.debug or self.fast_mode:
            h_o = (h_i + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            w_o = (w_i + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

            out_shape0 = (x.shape[0], in_channels, h_o, w_o)
            self._outshape = out_shape0

        if self.debug or not self.fast_mode:
            y = self.op(x)
            out_shape1 = y.shape
            self._outshape = tuple(out_shape1)

        if self.debug:
            assert tuple(out_shape1) == out_shape0

        filters_per_channel = 1
        active_elements_count = bs * np.prod(self._outshape[2:])
        conv_per_position_flops = (
            in_channels * filters_per_channel * np.prod(kernel_size)
        )

        overall_conv_flops = conv_per_position_flops * active_elements_count
        bias_flops = 0

        overall_flops = overall_conv_flops + bias_flops
        self._flops = overall_flops

        if self.fast_mode:
            return torch.zeros(out_shape0, device=x.device)
        else:
            return y


class AdaptiveAvgPool2dStats(OpStats):
    def __init__(self):
        super(AdaptiveAvgPool2dStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)
        output_size = self.op.output_size
        if output_size == 1:
            self._flops = np.prod(self._inshape)
            if self.debug or self.fast_mode:
                out_shape0 = list(x.shape)
                out_shape0[-1] = 1
                out_shape0[-2] = 1

                self._outshape = tuple(out_shape0)
            if self.debug or not self.fast_mode:
                y = self.op(x)
                out_shape1 = y.shape
                self._outshape = tuple(out_shape1)

            if self.debug:
                assert tuple(out_shape1) == out_shape0
        else:
            assert False

        if self.fast_mode:
            return torch.zeros(self._outshape, device=x.device)
        else:
            return y


class UpsampleStats(OpStats):
    def __init__(self):
        super(UpsampleStats, self).__init__()

    def forward(self, x: Tensor):
        in_shape = x.shape
        self._inshape = tuple(in_shape)
        upsample_size = self.op.size
        upsample_mode = self.op.mode

        if upsample_size is not None:
            h_o, w_o = upsample_size
            bs = in_shape[0]
            in_channels = in_shape[1]

            out_shape0 = (bs, in_channels, h_o, w_o)
            self._outshape = out_shape0

            if upsample_mode == "bilinear":
                self._flops = bs * in_channels * h_o * w_o * 6
            elif upsample_mode == "nearest":
                self._flops = bs * in_channels * h_o * w_o * 1
            else:
                assert False

            if self.debug or not self.fast_mode:
                y = self.op(x)
                out_shape1 = y.shape
                self._outshape = tuple(out_shape1)

            if self.debug:
                assert tuple(out_shape1) == out_shape0
        else:
            assert False

        if self.fast_mode:
            return torch.zeros(self._outshape, device=x.device)
        else:
            return y


MAPPINGS = {
    nn.Conv2d: Conv2dStats,
    nn.BatchNorm2d: BatchNorm2dStats,
    nn.ReLU6: ReLU6Stats,
    nn.ReLU: ReLU6Stats,
    nn.Dropout: DropoutStats,
    nn.Linear: LinearStats,
    nn.MaxPool2d: Pool2dStats,
    nn.AvgPool2d: Pool2dStats,
    nn.Flatten: FlattenStats,
    nn.Identity: IdentityStats,
    nn.AdaptiveAvgPool2d: AdaptiveAvgPool2dStats,
    nn.Upsample: UpsampleStats,
    nn.Conv1d: Conv1dStats,
}


class IPGraph:
    def __init__(self, model: GraphModule):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.fx_graph = self.model.graph
        self.nodes = list(self.fx_graph.nodes)

    def get_flops(self, input_shapes):
        device = next(self.model.parameters()).device
        named_modules = dict(self.model.named_modules())
        for node in self.model.graph.nodes:
            if node.op == "placeholder" or node.op == "output":
                pass
            elif node.op == "call_module":
                mod = named_modules[node.target]
                if type(mod) in MAPPINGS:
                    mockOp = MAPPINGS[type(mod)]()
                    mockOp.op = mod
                else:
                    mockOp = OpStats()
                    mockOp.op = named_modules[node.target]

                pname, sname = _parent_name(node.target)
                setattr(named_modules[pname], sname, mockOp)
            elif node.op == "call_function":
                pass
            elif node.op == "call_method":
                pass
            else:
                assert(False)

        self.model.recompile()
        self.model.graph.lint()

        inputs = []
        for input_shape in input_shapes:
            inputs.append(torch.rand(input_shape, device=device))
        inputs = tuple(inputs)

        self.model(*inputs)

        named_modules = dict(self.model.named_modules())
        flops_dict = {}
        for node in self.model.graph.nodes:
            if node.op == "placeholder" or node.op == "output":
                pass
            elif node.op == "call_module":
                mod = named_modules[node.target]
                flops = getattr(mod, "_flops", 0)
                flops_dict[node.name] = flops

            elif node.op == "call_function":
                pass
            else:
                pass

        return flops_dict
