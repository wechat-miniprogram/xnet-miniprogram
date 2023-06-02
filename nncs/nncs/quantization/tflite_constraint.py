from enum import Enum
import networkx as nx
from nncs.nn.intrinsic.custom_op import SharedFakeQuantize, MasterFakeQuantize
from nncs.fake_quantize.fake_quantize_base import FixedQParamsFakeQuantize


class ConstraintType(Enum):
    SHARED_FQ = 0
    MASTER_FQ = 1
    FIX_FQ = 2
    UNKNOWN = 3


Qparams = {
    "softmax": (1.0 / 256.0, -128),
    "sigmoid": (1.0 / 256.0, -128),
    "tanh": (1.0 / 128.0, 0),
    "log_softmax": (16.0 / 256.0, 127),
}

ConstraintOps = {
    "cat": ConstraintType.MASTER_FQ,
    "reshape": ConstraintType.SHARED_FQ,
    "view": ConstraintType.SHARED_FQ,
    "flatten": ConstraintType.SHARED_FQ,
    "squeeze": ConstraintType.SHARED_FQ,
    "unsqueeze": ConstraintType.SHARED_FQ,
    "transpose": ConstraintType.SHARED_FQ,
    "chunk": ConstraintType.SHARED_FQ,
    "permute": ConstraintType.SHARED_FQ,
    "adaptive_avg_pool2d": ConstraintType.SHARED_FQ,
    "avg_pool2d": ConstraintType.SHARED_FQ,
    "max_pool2d": ConstraintType.SHARED_FQ,
    "resize_bilinear": ConstraintType.SHARED_FQ,
    "interpolate": ConstraintType.SHARED_FQ,
    "softmax": ConstraintType.FIX_FQ,
    "sigmoid": ConstraintType.FIX_FQ,
    "tanh": ConstraintType.FIX_FQ,
    "__getitem__": ConstraintType.SHARED_FQ,
    "log_softmax": ConstraintType.FIX_FQ,
}

UnConstraintOps = {
    "conv1d": ConstraintType.UNKNOWN,
    "conv2d": ConstraintType.UNKNOWN,
    "nncs_model_input": ConstraintType.UNKNOWN,
    "conv1d_RELU": ConstraintType.UNKNOWN,
    "conv2d_RELU": ConstraintType.UNKNOWN,
    "linear_RELU": ConstraintType.UNKNOWN,
    "linear_hardtanh": ConstraintType.UNKNOWN,
    "conv2d_batch_norm_RELU": ConstraintType.UNKNOWN,
    "conv2d_batch_norm": ConstraintType.UNKNOWN,
    "conv2d_batch_norm_hardtanh": ConstraintType.UNKNOWN,
    "conv2d_hardtanh": ConstraintType.UNKNOWN,
    "conv2d_batch_norm_clamp": ConstraintType.UNKNOWN,
    "conv2d_clamp": ConstraintType.UNKNOWN,
    "batch_norm": ConstraintType.UNKNOWN,
    "conv_transpose2d_batch_norm": ConstraintType.UNKNOWN,
    "conv_transpose2d_batch_norm_RELU": ConstraintType.UNKNOWN,
    "conv_transpose2d_batch_norm_hardtanh": ConstraintType.UNKNOWN,
    "hardtanh": ConstraintType.UNKNOWN,
    "hardswish": ConstraintType.UNKNOWN,
    "hardsigmoid": ConstraintType.UNKNOWN,
    "leaky_relu": ConstraintType.UNKNOWN,
    "prelu": ConstraintType.UNKNOWN,
    "linear": ConstraintType.UNKNOWN,
    "__add__": ConstraintType.UNKNOWN,
    "__add___RELU": ConstraintType.UNKNOWN,
    "__add___clamp": ConstraintType.UNKNOWN,
    "__iadd__": ConstraintType.UNKNOWN,
    "__iadd___RELU": ConstraintType.UNKNOWN,
    "__radd__": ConstraintType.UNKNOWN,
    "__sub__": ConstraintType.UNKNOWN,
    "__isub__": ConstraintType.UNKNOWN,
    "__rsub__": ConstraintType.UNKNOWN,
    "mul": ConstraintType.UNKNOWN,
    "__mul__": ConstraintType.UNKNOWN,
    "__mul___RELU": ConstraintType.UNKNOWN,
    "__mul___hardtanh": ConstraintType.UNKNOWN,
    "__mul___clamp": ConstraintType.UNKNOWN,
    "__imul__": ConstraintType.UNKNOWN,
    "__rmul__": ConstraintType.UNKNOWN,
    "__div__": ConstraintType.UNKNOWN,
    "__idiv__": ConstraintType.UNKNOWN,
    "__truediv__": ConstraintType.UNKNOWN,
    "__setitem__": ConstraintType.UNKNOWN,
    "multiply": ConstraintType.UNKNOWN,
    "layer_norm": ConstraintType.UNKNOWN,
    "log": ConstraintType.UNKNOWN,
    "matmul": ConstraintType.UNKNOWN,
    "sum": ConstraintType.UNKNOWN,
    "add": ConstraintType.UNKNOWN,
    "mean": ConstraintType.UNKNOWN,
}


class Constraint:
    def __init__(self, op_name, constraint_type, node_list=None):
        self.op_name = op_name
        self.constraint_type = constraint_type
        if node_list is None:
            self.node_list = []

    def apply_to(self, module, graph):
        if ConstraintType.SHARED_FQ == self.constraint_type:
            op_name = graph.nodes[self.node_list[0]]["op_exec_context"].operator_name
            fq = module.activation_quantizers[self.node_list[1]]
            if (
                op_name in ConstraintOps
                and ConstraintType.MASTER_FQ == ConstraintOps[op_name]
            ):
                if not isinstance(fq, MasterFakeQuantize):
                    assert False

            if (
                op_name in ConstraintOps
                and ConstraintType.FIX_FQ == ConstraintOps[op_name]
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
            cond_name = []
            for i in range(1, len(self.node_list)):
                op_name = graph.nodes[self.node_list[i]][
                    "op_exec_context"
                ].operator_name
                if (
                    op_name in ConstraintOps
                    and ConstraintType.FIX_FQ == ConstraintOps[op_name]
                ):
                    cond_name.append(op_name)

            if len(cond_name) == 0:
                master_fq = MasterFakeQuantize()
                fq = module.activation_quantizers[self.node_list[0]]
                master_fq.master_fq = fq
                master_fq.fq_list = []
                for i in range(1, len(self.node_list)):
                    fq = module.activation_quantizers[self.node_list[i]]
                    master_fq.fq_list.append(fq)

                module.activation_quantizers[self.node_list[0]] = master_fq
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
            fq = module.activation_quantizers[self.node_list[0]]
            quant_min = fq.quant_min
            quant_max = fq.quant_max
            fixed_fq = FixedQParamsFakeQuantize(
                scale=Qparams[self.op_name][0],
                zero_point=Qparams[self.op_name][1],
                dtype=fq.dtype,
                qscheme=fq.qscheme,
                quant_min=quant_min,
                quant_max=quant_max,
            )
            module.activation_quantizers[self.node_list[0]] = fixed_fq
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
        predecessors = list(graph.predecessors(node["key"]))

        for predecessor in predecessors:
            p = graph.nodes[predecessor]
            op_name = p["op_exec_context"].operator_name
            if op_name in ConstraintOps:
                if ConstraintType.SHARED_FQ == ConstraintOps[op_name]:
                    self.propagate(graph, p, c)
                else:
                    c.node_list.append(predecessor)
            else:
                c.node_list.append(predecessor)

    def apply_to(self, nxgraph):
        graph = nxgraph
        weakly_subgraphs = [
            graph.subgraph(c) for c in nx.weakly_connected_components(graph)
        ]
        for subgraph in weakly_subgraphs:
            dfs_order = nx.topological_sort(subgraph)
            for node in dfs_order:
                if "nncs_no_quant" in node:
                    continue
                op_name = graph.nodes[node]["op_exec_context"].operator_name
                if op_name in ConstraintOps:
                    self.constriant_nodes.append(node)
                elif op_name in UnConstraintOps:
                    pass
                else:
                    assert False, "Unsupport Constraint Op {}".format(op_name)

        while len(self.constriant_nodes) != 0:
            cur_node_name = self.constriant_nodes.pop()
            cur_node = graph.nodes[cur_node_name]
            op_name = cur_node["op_exec_context"].operator_name
            c = Constraint(op_name, ConstraintType.UNKNOWN)
            c.node_list.append(cur_node_name)

            if op_name in ConstraintOps:
                constraint_type = ConstraintOps[op_name]
                c.constraint_type = constraint_type
                if ConstraintType.MASTER_FQ == constraint_type:
                    self.propagate(graph, cur_node, c)
                elif ConstraintType.SHARED_FQ == ConstraintOps[op_name]:
                    self.propagate(graph, cur_node, c)
                elif ConstraintType.FIX_FQ == ConstraintOps[op_name]:
                    pass
                else:
                    assert False, "ConstraintType Unknown {}".format(op_name)
            else:
                assert False, "Not a ConstraintOps"

            self.cs.append(c)
