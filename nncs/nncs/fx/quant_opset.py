import operator
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.googlenet import GoogLeNetOutputs
from torchvision.models.inception import InceptionOutputs
from torchvision.ops import StochasticDepth, stochastic_depth

from nncs.fx.no_quant import FXNNCSNoQuant
import nncs.nn.qat as nnqat
import nncs.nn.intrinsic.qat as nniqat
from nncs.fx.extra_fusion_patterns import (
    BinaryClip,
    SharedOpDropout,
    RelayoutContiguousOp,
    RelayoutFunctionContiguousOp,
)


onnx_ignore_op_set = set(
    [
        "contiguous",
        "clone",
        "detach",
        nn.Dropout,
        StochasticDepth,
    ]
)

input_op_set = set(["nncs_input"])

quant_op_set = set(
    [
        nnqat.Conv1d,
        nniqat.ConvReLU2d,
        nniqat.ConvReLU62d,
        nniqat.LinearBn1d,
        nniqat.LinearReLU,
        nnqat.Linear,
        nniqat.ConvBnReLU62d,
        nniqat.ConvBnReLU2d,
        nniqat.ConvBn2d,
        nnqat.Conv2d,
        BinaryClip,
        nniqat.ConvLearnableClip82d,
        nniqat.ConvTransposeBnReLU2d,
        nniqat.ConvTransposeBnReLU62d,
        nniqat.ConvTransposeReLU2d,
        nniqat.ConvTransposeReLU62d,
        nnqat.ConvTranspose2d,
    ]
)

binary_op_set = set(
    [
        'add',
        operator.add,
        operator.mul,
        operator.truediv,
        operator.sub,
    ]
)

optional_relayout_quant_op_set = set(
    [
        nn.Flatten,
        "flatten",
        torch.flatten,
        "view",
        "reshape",
        torch.reshape,
        "permute",
        torch.permute,
        "transpose",
        torch.transpose,
        "unsqueeze",
        torch.unsqueeze,
        "squeeze",
        torch.squeeze,
        "contiguous",
        "chunk",
        SharedOpDropout,
        nn.MaxPool2d,
        F.max_pool2d,
        nn.AvgPool2d,
        F.avg_pool2d,
        nn.AdaptiveAvgPool2d,
        F.adaptive_avg_pool2d,
        F.interpolate,
        nn.Upsample,
        operator.getitem,
        RelayoutContiguousOp,
        RelayoutFunctionContiguousOp,
        nn.Dropout,
        stochastic_depth,
        StochasticDepth,
        "mean",
        "clone",
        "detach",
    ]
)

optional_activation_quant_op_set = set(
    [
        nn.Softmax,
        torch.softmax,
        F.softmax,
        torch.log_softmax,
        F.log_softmax,
        torch.clip,
        torch.tanh,
        "sigmoid",
        F.sigmoid,
        torch.sigmoid,
        nn.Sigmoid,
    ]
)

optional_cat_quant_op_set = set([torch.cat])

optional_quant_op_set = (
    optional_relayout_quant_op_set
    | optional_activation_quant_op_set
    | optional_cat_quant_op_set
)

not_quant_op_set = set(
    [
        "nncs_constant",
        F.pad,
        torch.matmul,
        nn.LayerNorm,
        torch.pow,
        nn.PReLU,
        nn.PixelShuffle,
        nn.Hardswish,
        nn.Hardsigmoid,
        F.hardsigmoid,
        nn.LeakyReLU,
        nn.SiLU,
        FXNNCSNoQuant,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.ReLU6,
        F.relu,
        F.relu6,
        GoogLeNetOutputs,
        InceptionOutputs,
        operator.neg,
        torch.multiply,
        nn.MultiheadAttention,
    ]
)


def is_quant_op(fx_node, model, nx_graph, no_quant_nodes, results):
    if isinstance(fx_node, torch.fx.node.Node):
        if fx_node.name in no_quant_nodes:
            return False

        if fx_node.name not in nx_graph.nodes:
            return False

        node_type = nx_graph.nodes[fx_node.name]["node_type"]
        if node_type not in ["quant"]:
            return False

        op_type = nx_graph.nodes[fx_node.name]["op_type"]

        if op_type in not_quant_op_set:
            return False
        elif op_type in quant_op_set:
            predecessors = list(nx_graph.predecessors(fx_node.name))
            for predecessor in predecessors:
                node_type = nx_graph.nodes[predecessor]["node_type"]
                if node_type in ["quant"]:
                    results[predecessor] = True
            return True
        elif op_type in input_op_set:
            return False
        elif op_type in binary_op_set:
            flags = []
            assert len(fx_node.args) == 2
            arg0 = fx_node.args[0]
            arg1 = fx_node.args[1]

            if isinstance(arg0, torch.fx.node.Node) and isinstance(
                arg1, torch.fx.node.Node
            ):
                flags.append(results[arg0.name])
                flags.append(results[arg1.name])

                if False not in flags:
                    return True
                else:
                    return False
            else:
                return False
        elif op_type in optional_relayout_quant_op_set:
            pre_flags = []
            pre_flags.append(results[fx_node.all_input_nodes[0].name])

            if False not in pre_flags:
                return True

            return False
        elif op_type in optional_quant_op_set:
            flags = []
            if op_type in optional_activation_quant_op_set:
                assert len(fx_node.all_input_nodes) == 1
                flags.append(results[fx_node.all_input_nodes[0].name])
            else:
                for arg in fx_node.all_input_nodes:
                    flags.append(results[arg.name])
            if False not in flags:
                return True
            else:
                return False
        else:
            print(op_type)
            import ipdb

            ipdb.set_trace()
            assert False
    elif isinstance(fx_node, (int, float)):
        return False
    else:
        assert False
