from enum import Enum
import operator
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth, stochastic_depth

import nncs.nn.qat as nnqat
import nncs.nn.intrinsic.qat as nniqat
from nncs.nn.intrinsic.custom_op import SharedFakeQuantize, MasterFakeQuantize
from nncs.fake_quantize.fake_quantize_base import FixedQParamsFakeQuantize

from nncs.fx.extra_fusion_patterns import (
    BinaryClip,
    SharedOpDropout,
    RelayoutContiguousOp,
    RelayoutFunctionContiguousOp,
)


class ConstraintType(Enum):
    SHARED_FQ = 0
    MASTER_FQ = 1
    FIX_FQ = 2
    UNKNOWN = 3


Qparams = {
    nn.Softmax: (1.0 / 256.0, -128),
    F.softmax: (1.0 / 256.0, -128),
    "tanh": (1.0 / 128.0, 0),
    torch.tanh: (1.0 / 128.0, 0),
    F.log_softmax: (16.0 / 256.0, 127),
    "sigmoid": (1.0 / 256.0, -128),
    nn.Sigmoid: (1.0 / 256.0, -128),
    F.sigmoid: (1.0 / 256.0, -128),
}

ConstraintOps = {
    "view": ConstraintType.SHARED_FQ,
    "reshape": ConstraintType.SHARED_FQ,
    "transpose": ConstraintType.SHARED_FQ,
    "permute": ConstraintType.SHARED_FQ,
    torch.transpose: ConstraintType.SHARED_FQ,
    "flatten": ConstraintType.SHARED_FQ,
    torch.flatten: ConstraintType.SHARED_FQ,
    nn.Flatten: ConstraintType.SHARED_FQ,
    "squeeze": ConstraintType.SHARED_FQ,
    F.log_softmax: ConstraintType.FIX_FQ,
    torch.cat: ConstraintType.MASTER_FQ,
    nn.AdaptiveAvgPool2d: ConstraintType.SHARED_FQ,
    F.adaptive_avg_pool2d: ConstraintType.SHARED_FQ,
    "chunk": ConstraintType.SHARED_FQ,
    nn.Identity: ConstraintType.SHARED_FQ,
    F.interpolate: ConstraintType.SHARED_FQ,
    nn.Upsample: ConstraintType.SHARED_FQ,
    F.softmax: ConstraintType.FIX_FQ,
    nn.Softmax: ConstraintType.FIX_FQ,
    nn.Sigmoid: ConstraintType.FIX_FQ,
    operator.getitem: ConstraintType.SHARED_FQ,
    SharedOpDropout: ConstraintType.SHARED_FQ,
    "unsqueeze": ConstraintType.SHARED_FQ,
    torch.unsqueeze: ConstraintType.SHARED_FQ,
    torch.tanh: ConstraintType.FIX_FQ,
    "contiguous": ConstraintType.SHARED_FQ,
    RelayoutContiguousOp: ConstraintType.SHARED_FQ,
    RelayoutFunctionContiguousOp: ConstraintType.SHARED_FQ,
    nn.Dropout: ConstraintType.SHARED_FQ,
    nn.MaxPool2d: ConstraintType.SHARED_FQ,
    F.max_pool2d: ConstraintType.SHARED_FQ,
    nn.AvgPool2d: ConstraintType.SHARED_FQ,
    F.avg_pool2d: ConstraintType.SHARED_FQ,
    stochastic_depth: ConstraintType.SHARED_FQ,
    StochasticDepth: ConstraintType.SHARED_FQ,
    # __getitem__: ConstraintType.SHARED_FQ,
}

UnConstraintOps = {
    "nncs_input": ConstraintType.UNKNOWN,
    nniqat.ConvBnReLU62d: ConstraintType.UNKNOWN,
    nniqat.ConvBnReLU2d: ConstraintType.UNKNOWN,
    nniqat.ConvBn2d: ConstraintType.UNKNOWN,
    nnqat.Conv2d: ConstraintType.UNKNOWN,
    nnqat.Conv1d: ConstraintType.UNKNOWN,
    nniqat.ConvReLU2d: ConstraintType.UNKNOWN,
    nniqat.ConvReLU62d: ConstraintType.UNKNOWN,
    nniqat.LinearBn1d: ConstraintType.UNKNOWN,
    nniqat.LinearReLU: ConstraintType.UNKNOWN,
    BinaryClip: ConstraintType.UNKNOWN,
    'add': ConstraintType.UNKNOWN,
    operator.add: ConstraintType.UNKNOWN,
    operator.mul: ConstraintType.UNKNOWN,
    nnqat.Linear: ConstraintType.UNKNOWN,
    nn.PReLU: ConstraintType.UNKNOWN,
    nniqat.ConvLearnableClip82d: ConstraintType.UNKNOWN,
    nn.Hardswish: ConstraintType.UNKNOWN,
    nn.Hardsigmoid: ConstraintType.UNKNOWN,
    F.hardsigmoid: ConstraintType.UNKNOWN,
    nn.LeakyReLU: ConstraintType.UNKNOWN,
    torch.matmul: ConstraintType.UNKNOWN,
    nniqat.ConvTransposeBnReLU2d: ConstraintType.UNKNOWN,
    nniqat.ConvTransposeBnReLU62d: ConstraintType.UNKNOWN,
    nniqat.ConvTransposeReLU2d: ConstraintType.UNKNOWN,
    nniqat.ConvTransposeReLU62d: ConstraintType.UNKNOWN,
    nnqat.ConvTranspose2d: ConstraintType.UNKNOWN,
    nn.SiLU: ConstraintType.UNKNOWN,
    "nncs_constant": ConstraintType.UNKNOWN,
    operator.sub: ConstraintType.UNKNOWN,
    F.pad: ConstraintType.UNKNOWN,
    nn.LayerNorm: ConstraintType.UNKNOWN,
    torch.pow: ConstraintType.UNKNOWN,
    operator.truediv: ConstraintType.UNKNOWN,
    nn.BatchNorm2d: ConstraintType.UNKNOWN,
    nn.ReLU: ConstraintType.UNKNOWN,
    nn.ReLU6: ConstraintType.UNKNOWN,
    F.relu: ConstraintType.UNKNOWN,
    F.relu6: ConstraintType.UNKNOWN,
    "mean": ConstraintType.UNKNOWN,
    nn.MultiheadAttention: ConstraintType.UNKNOWN,
}


class Constraint:
    def __init__(self, op_name, constraint_type, node_list=None):
        self.op_name = op_name
        self.constraint_type = constraint_type
        if node_list is None:
            self.node_list = []

    def apply_to(self, module, graph, quant_markers):
        if ConstraintType.SHARED_FQ == self.constraint_type:
            if not quant_markers[self.node_list[1]]:
                return None

            op_type = graph.nodes[self.node_list[0]]["op_type"]
            fq = module.activation_quantizers[self.node_list[1]]
            if (
                op_type in ConstraintOps
                and ConstraintType.MASTER_FQ == ConstraintOps[op_type]
            ):
                if not isinstance(fq, MasterFakeQuantize):
                    assert False

            if (
                op_type in ConstraintOps
                and ConstraintType.FIX_FQ == ConstraintOps[op_type]
            ):
                if not isinstance(fq, FixedQParamsFakeQuantize):
                    assert False

            if isinstance(fq, MasterFakeQuantize):
                _fq = fq.master_fq
            else:
                _fq = fq
            shared_fq = SharedFakeQuantize(_fq)
            return shared_fq
        elif ConstraintType.MASTER_FQ == self.constraint_type:
            flags = []
            for i in range(1, len(self.node_list)):
                flags.append(quant_markers[self.node_list[i]])

            if False in flags:
                return None

            cond_name = []
            for i in range(1, len(self.node_list)):
                op_type = graph.nodes[self.node_list[i]]["op_type"]
                if (
                    op_type in ConstraintOps
                    and ConstraintType.FIX_FQ == ConstraintOps[op_type]
                ):
                    cond_name.append(op_type)

            if len(cond_name) == 0:
                master_fq = MasterFakeQuantize()
                fq = module.activation_quantizers[self.node_list[0]]
                master_fq.master_fq = fq
                master_fq.fq_list = []
                for i in range(1, len(self.node_list)):
                    fq = module.activation_quantizers[self.node_list[i]]
                    master_fq.fq_list.append(fq)

                return master_fq
            elif len(cond_name) == 1:
                master_fq = MasterFakeQuantize()
                fq = module.activation_quantizers[self.node_list[0]]
                quant_min = fq.quant_min
                quant_max = fq.quant_max
                cond_op = cond_name[0]
                fixed_fq = FixedQParamsFakeQuantize(
                    scale=Qparams[cond_op][0],
                    zero_point=Qparams[cond_op][1],
                    dtype=fq.dtype,
                    qscheme=fq.qscheme,
                    quant_min=quant_min,
                    quant_max=quant_max,
                )
                master_fq.master_fq = fixed_fq
                master_fq.fq_list = []
                for i in range(1, len(self.node_list)):
                    fq = module.activation_quantizers[self.node_list[i]]
                    master_fq.fq_list.append(fq)

                return master_fq
            elif len(cond_name) >= 2:
                qparams = Qparams[cond_name[0]]
                flag = True
                for i in range(1, len(cond_name)):
                    _qparams = Qparams[cond_name[i]]
                    if qparams != _qparams:
                        flag = False

                if flag:
                    master_fq = MasterFakeQuantize()
                    fq = module.activation_quantizers[self.node_list[0]]
                    quant_min = fq.quant_min
                    quant_max = fq.quant_max
                    cond_op = cond_name[0]
                    fixed_fq = FixedQParamsFakeQuantize(
                        scale=Qparams[cond_op][0],
                        zero_point=Qparams[cond_op][1],
                        dtype=fq.dtype,
                        qscheme=fq.qscheme,
                        quant_min=quant_min,
                        quant_max=quant_max,
                    )
                    master_fq.master_fq = fixed_fq
                    master_fq.fq_list = []
                    for i in range(1, len(self.node_list)):
                        fq = module.activation_quantizers[self.node_list[i]]
                        master_fq.fq_list.append(fq)
                    return master_fq
                else:
                    # keep normal fq
                    fq = module.activation_quantizers[self.node_list[0]]
                    return fq

        elif ConstraintType.FIX_FQ == self.constraint_type:
            op_type = graph.nodes[self.node_list[0]]["op_type"]
            fq = module.activation_quantizers[self.node_list[0]]
            quant_min = fq.quant_min
            quant_max = fq.quant_max
            fixed_fq = FixedQParamsFakeQuantize(
                scale=Qparams[op_type][0],
                zero_point=Qparams[op_type][1],
                dtype=fq.dtype,
                qscheme=fq.qscheme,
                quant_min=quant_min,
                quant_max=quant_max,
            )
            return fixed_fq
        else:
            assert False, "Unknown ConstraintType {}".format(self.constraint_type)

    def __str__(self):
        out_str = "{}: {}".format(
            self.constraint_type, "|".join([p for p in self.node_list])
        )
        return out_str


class ConstraintSolver:
    def __init__(self):
        self.constriant_nodes = []
        self.cs = []

    def propagate(self, graph, node, c):
        predecessors = list(graph.predecessors(node))
        for predecessor in predecessors:
            p = graph.nodes[predecessor]
            if p["node_type"] == "attr":
                continue
            op_type = p["op_type"]
            if op_type in ConstraintOps:
                if ConstraintType.SHARED_FQ == ConstraintOps[op_type]:
                    self.propagate(graph, predecessor, c)
                else:
                    c.node_list.append(predecessor)
            else:
                c.node_list.append(predecessor)

    def apply_to(self, nx_graph, no_quant_nodes, quant_markers):
        graph = nx_graph
        weakly_subgraphs = [
            graph.subgraph(c) for c in nx.weakly_connected_components(graph)
        ]
        for subgraph in weakly_subgraphs:
            dfs_order = nx.topological_sort(subgraph)
            for node in dfs_order:

                if node in no_quant_nodes:
                    continue

                node_type = graph.nodes[node]["node_type"]
                if node_type in ["attr"]:
                    continue

                if quant_markers is not None:
                    fx_node_name = graph.nodes[node]["fx_node_name"]
                    if not quant_markers[fx_node_name]:
                        continue

                op_type = graph.nodes[node]["op_type"]
                if op_type in ConstraintOps:
                    self.constriant_nodes.append(node)
                elif op_type in UnConstraintOps:
                    pass
                else:
                    assert False, "Unsupport Constraint Op {}".format(op_type)

        while len(self.constriant_nodes) != 0:
            cur_node_name = self.constriant_nodes.pop()
            cur_node = graph.nodes[cur_node_name]
            op_type = cur_node["op_type"]
            c = Constraint(cur_node_name, ConstraintType.UNKNOWN)
            c.node_list.append(cur_node_name)

            if op_type in ConstraintOps:
                constraint_type = ConstraintOps[op_type]
                c.constraint_type = constraint_type
                if ConstraintType.MASTER_FQ == constraint_type:
                    self.propagate(graph, cur_node_name, c)
                elif ConstraintType.SHARED_FQ == ConstraintOps[op_type]:
                    self.propagate(graph, cur_node_name, c)
                elif ConstraintType.FIX_FQ == ConstraintOps[op_type]:
                    pass
                else:
                    assert False, "ConstraintType Unknown {}".format(op_type)
            else:
                assert False, "Not a ConstraintOps"

            self.cs.append(c)

    def get_constraint_by_name(self, node_name):
        for cs in self.cs:
            if node_name == cs.op_name:
                return cs
        return None
