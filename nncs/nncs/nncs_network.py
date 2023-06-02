import inspect
from collections import Counter
from typing import List, Callable, Optional
from enum import Enum
from copy import deepcopy
import functools

import networkx as nx
import torch
from torch import nn

from nncs.dynamic_graph.graph import (
    NNCSGraph,
    InputAgnosticOperationExecutionContext,
    OperationExecutionContext,
)
from nncs.dynamic_graph.graph_builder import (
    GraphBuilder,
    create_dummy_forward_fn,
    ModelInputInfo,
)
from nncs.dynamic_graph.graph_matching import NodeExpression, search_all
from nncs.dynamic_graph.input_wrapping import MODEL_INPUT_OP_NAME, InputInfoWrapManager
from nncs.dynamic_graph.wrappers import ignore_scope
from nncs.dynamic_graph.context import TracingContext
from nncs.utils import (
    get_module_by_node_name,
    set_module_by_node_name,
    get_all_modules,
    get_mname_by_nname,
    training_mode_switcher,
    get_module,
)
from nncs.quantization.tflite_constraint import (
    ConstraintSolver as TFConstraintSolver,
)


class ExtraCompressionModuleType(Enum):
    ACTIVATION_QUANTIZER = 0


class MergedOpGraph(nx.DiGraph):
    def __init__(self, model_nx_graph: nx.DiGraph):
        super().__init__()
        self._base_nx_graph = deepcopy(model_nx_graph)
        self._input_ips = []

    def _base_graph_match_has_breaking_output_edges(self, match):
        for node_key in match[:-1]:
            succs = list(self._base_nx_graph.succ[node_key].keys())
            for succ_key in succs:
                if succ_key not in match:
                    return True
        return False

    def get_graph_with_merged_operations(self):
        merged_graph = deepcopy(self._base_nx_graph)
        pattern = self._get_mergeable_operator_patterns()
        matches = search_all(self._base_nx_graph, pattern)
        self.matches = []
        for match in matches:
            if len(match) == 1:
                continue

            flag = False
            for m in match:
                if "NNCSNoQuant" in m:
                    flag = True
            if flag:
                continue

            input_node_key = match[0]
            output_node_key = match[-1]

            # If a subgraph has output edges in its middle, should skip merging it
            # Example:
            #       (conv2d)
            #          |------\
            #         (BN)    |
            #          |      |
            #        (RELU)   |
            #          |      |
            #        (cat)----/
            #          |
            #         ...

            has_breaking_output_edges = (
                self._base_graph_match_has_breaking_output_edges(match)
            )

            if has_breaking_output_edges:
                continue

            self.matches.append(match)

            in_edges = list(merged_graph.in_edges(input_node_key))
            out_edges = list(merged_graph.out_edges(output_node_key))

            in_edge_copies_dict = {}
            for in_edge_key in in_edges:
                in_edge_copies_dict[in_edge_key] = deepcopy(
                    merged_graph.edges[in_edge_key]
                )

            out_edge_copies_dict = {}
            for out_edge_key in out_edges:
                out_edge_copies_dict[out_edge_key] = deepcopy(
                    merged_graph.edges[out_edge_key]
                )

            # conserved_edges_list = out_edges + in_edges

            merged_node_attrs = deepcopy(merged_graph.nodes[input_node_key])
            # merged_node_key = ""
            merged_node_key = []
            operator_names = []
            for node_key in match:
                # merged_node_key += node_key + '\n'
                merged_node_key.append(node_key)
                operator_names.append(
                    merged_graph.nodes[node_key][
                        "op_exec_context"
                    ].input_agnostic.operator_name
                )
                merged_graph.remove_node(node_key)

            merged_node_key = "\n".join(merged_node_key)
            merged_node_attrs[
                "op_exec_context"
            ].input_agnostic.operator_name = "_".join(operator_names)
            merged_graph.add_node(merged_node_key, **merged_node_attrs)
            for in_edge_key, in_edge_attrs in in_edge_copies_dict.items():
                merged_graph.add_edge(in_edge_key[0], merged_node_key, **in_edge_attrs)
            for out_edge_key, out_edge_attrs in out_edge_copies_dict.items():
                merged_graph.add_edge(
                    merged_node_key, out_edge_key[1], **out_edge_attrs
                )

        return merged_graph

    def _get_mergeable_operator_patterns(self) -> NodeExpression:
        import nncs.dynamic_graph.patterns as p

        pattern = (
            p.LINEAR_OPS + p.ANY_BN_ACT_COMBO
            | p.LINEAR_OPS + p.ELTWISE_UNIFORM_OPS
            | p.ARITHMETIC + p.ANY_BN_ACT_COMBO
            | p.ANY_BN_ACT_COMBO
        )
        return pattern


@functools.total_ordering
class OperationPriority(Enum):
    DEFAULT_PRIORITY = 0
    FP32_TENSOR_STATISTICS_OBSERVATION = 1
    SPARSIFICATION_PRIORITY = 2
    QUANTIZATION_PRIORITY = 11
    PRUNING_PRIORITY = 1

    def __lt__(self, other):
        # pylint: disable=comparison-with-callable
        return self.value < other.value


class InsertionType(Enum):
    OPERATOR_PRE_HOOK = 0
    OPERATOR_POST_HOOK = 1
    NNCS_MODULE_PRE_OP = 2
    NNCS_MODULE_POST_OP = 3

    def __eq__(self, other):
        # pylint: disable=comparison-with-callable
        if isinstance(other, InsertionType):
            return self.value == other.value
        return self.value == other


class InsertionInfo:
    def __init__(
        self,
        op_exec_context: OperationExecutionContext,
        in_port_id: Optional[int] = None,
        is_input=False,
        is_output=False,
        shape_to_operate_on=None,
    ):
        self.op_exec_context = op_exec_context  # type: OperationExecutionContext
        self.in_port_id = (
            in_port_id  # None for post-hook quantization, otherwise - pre-hook
        )
        self.is_input = is_input
        self.is_output = is_output
        self.shape_to_operate_on = shape_to_operate_on
        self._linked_insertion_infos = []  # type: List[InsertionInfo]

    def get_linked_insertion_infos(self) -> List["InsertionInfo"]:
        return self._linked_insertion_infos

    def link_insertion_infos(self, linked_insertion_infos: List["InsertionInfo"]):
        self._linked_insertion_infos += linked_insertion_infos

    def __eq__(self, other: "InsertionInfo"):
        # TODO: ensure no circular refs via self._linked_insertion_infos?
        return (
            self.op_exec_context == other.op_exec_context
            and Counter(self._linked_insertion_infos)
            == Counter(other.get_linked_insertion_infos())
            and self.in_port_id == other.in_port_id
        )

    def __str__(self):
        postfix = ""
        if self.in_port_id is not None:
            postfix = "|INPUT{}".format(self.in_port_id)
        return str(self.op_exec_context.input_agnostic) + postfix

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def from_insertion_point(cls, ip: "InsertionPoint") -> "InsertionInfo":
        return cls(
            OperationExecutionContext(
                operator_name=ip.ia_op_exec_context.operator_name,
                scope_in_model=ip.ia_op_exec_context.scope_in_model,
                call_order=ip.ia_op_exec_context.call_order,
                tensor_metas=[None],
            ),
            in_port_id=ip.input_port_id,
        )


class InsertionPoint:
    def __init__(
        self,
        insertion_type: InsertionType,
        *,
        ia_op_exec_context: InputAgnosticOperationExecutionContext = None,
        module_scope: "Scope" = None,
        input_port_id: int = None
    ):
        self.insertion_type = insertion_type
        if self.insertion_type in [
            InsertionType.NNCS_MODULE_PRE_OP,
            InsertionType.NNCS_MODULE_POST_OP,
        ]:
            if module_scope is None:
                raise ValueError(
                    "Should specify module scope for module pre- and post-op insertion points!"
                )

        if self.insertion_type in [
            InsertionType.OPERATOR_PRE_HOOK,
            InsertionType.OPERATOR_POST_HOOK,
        ]:
            if ia_op_exec_context is None:
                raise ValueError(
                    "Should specify an operator's InputAgnosticOperationExecutionContext "
                    "for operator pre- and post-hook insertion points!"
                )
        self.module_scope = module_scope
        self.ia_op_exec_context = ia_op_exec_context
        self.input_port_id = input_port_id

    def __eq__(self, other: "InsertionPoint"):
        return (
            self.insertion_type == other.insertion_type
            and self.ia_op_exec_context == other.ia_op_exec_context
            and self.input_port_id == other.input_port_id
            and self.module_scope == other.module_scope
        )

    def __str__(self):
        prefix = str(self.insertion_type)
        retval = prefix
        if self.insertion_type in [
            InsertionType.NNCS_MODULE_PRE_OP,
            InsertionType.NNCS_MODULE_POST_OP,
        ]:
            retval += " {}".format(self.module_scope)
        elif self.insertion_type in [
            InsertionType.OPERATOR_PRE_HOOK,
            InsertionType.OPERATOR_POST_HOOK,
        ]:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.ia_op_exec_context)
        return retval

    def __hash__(self):
        return hash(str(self))


class InsertionCommand:
    def __init__(
        self,
        point: InsertionPoint,
        fn: Callable,
        priority: OperationPriority = OperationPriority.DEFAULT_PRIORITY,
    ):
        self.insertion_point = point  # type: InsertionPoint
        self.fn = fn  # type: Callable
        self.priority = priority  # type: OperationPriority


class InsertionPointGraphNodeType(Enum):
    INSERTION_POINT = 0
    OPERATOR = 1


class InsertionPointGraph(nx.DiGraph):
    NODE_TYPE_NODE_ATTR = "node_type"
    INSERTION_POINT_DATA_NODE_ATTR = "insertion_point_data"
    IS_IN_NNCS_MODULE_NODE_ATTR = "is_in_nncs_module"
    REGULAR_NODE_REF_NODE_ATTR = "regular_node_ref"
    ASSOCIATED_IP_NODE_KEYS_NODE_ATTR = "associated_ip_node_keys"
    OPERATOR_METATYPE_NODE_ATTR = "op_meta"

    PRE_HOOK_ID_PREFIX = "PRE HOOK "  # NB: Do not use colon (':') in node keys! Causes trouble for .dot file export.
    POST_HOOK_ID_PREFIX = "POST HOOK "

    def __init__(self, model_nx_graph: nx.DiGraph):
        super().__init__()
        self._base_nx_graph = deepcopy(model_nx_graph)
        self._input_ips = []  # type: List[InsertionPoint]

        for node_key, node in self._base_nx_graph.nodes.items():
            attrs = {
                InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR: node,
                InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.OPERATOR,
                InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR: set(),
                InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR: None,
            }
            self.add_node(node_key, **attrs)

        IN_PORT_ID_ATTR_NAME = "in_port_id"
        for edge in self._base_nx_graph.edges:
            in_port_id = self._base_nx_graph.edges[edge][
                NNCSGraph.IN_PORT_NAME_EDGE_ATTR
            ]
            from_node, to_node = edge
            attrs = {IN_PORT_ID_ATTR_NAME: in_port_id}
            self.add_edge(from_node, to_node, **attrs)

        node_keys_working_set = [deepcopy(node_key) for node_key in self.nodes.keys()]
        for operator_node_key in node_keys_working_set:
            original_node = self.nodes[operator_node_key][
                InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR
            ]
            ia_op_exec_context = original_node[
                NNCSGraph.OP_EXEC_CONTEXT_NODE_ATTR
            ].input_agnostic

            # Pre-hook insertion point nodes
            # Will insert a pre-hook IP for each input edge. The input edge must be marked with
            # a port ID attribute.
            # operator_node = self.nodes[operator_node_key]
            # in_edges = list(self.in_edges(operator_node_key))
            # for edge in in_edges:
            #    port_id = self.edges[edge][IN_PORT_ID_ATTR_NAME]
            #    from_node_key, to_node_key = edge
            #    ip_node_key = self.get_pre_hook_node_key(str(operator_node_key), port_id)

            #    pre_hook_insertion_point = InsertionPoint(InsertionType.OPERATOR_PRE_HOOK,
            #                                              ia_op_exec_context=ia_op_exec_context,
            #                                              input_port_id=port_id)
            #    pre_hook_ip_attrs = {
            #        InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.INSERTION_POINT,
            #        InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR: pre_hook_insertion_point,
            #    }

            #    self.add_node(ip_node_key, **pre_hook_ip_attrs)

            #    self.remove_edge(from_node_key, to_node_key)
            #    self.add_edge(from_node_key, ip_node_key)
            #    self.add_edge(ip_node_key, operator_node_key)
            #    operator_node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(ip_node_key)

            # Post-hook insertion point nodes
            post_hook_insertion_point = InsertionPoint(
                InsertionType.OPERATOR_POST_HOOK, ia_op_exec_context=ia_op_exec_context
            )
            post_hook_ip_attrs = {
                InsertionPointGraph.NODE_TYPE_NODE_ATTR: InsertionPointGraphNodeType.INSERTION_POINT,
                InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR: post_hook_insertion_point,
            }
            ip_node_key = self.get_post_hook_node_key(str(operator_node_key))
            self.add_node(ip_node_key, **post_hook_ip_attrs)
            out_edges = list(self.out_edges(operator_node_key))
            for out_edge in out_edges:
                # Need to preserve original edge attributes in order not to lose
                # input port ID information
                original_edge_attrs = self.edges[out_edge]
                from_node_key, to_node_key = out_edge
                self.remove_edge(from_node_key, to_node_key)
                self.add_edge(ip_node_key, to_node_key, **original_edge_attrs)
                # TODO: introduce separate insertion points for operator outputs if
                # the outputs are semantically different
            self.add_edge(operator_node_key, ip_node_key)
            operator_node = self.nodes[operator_node_key]
            operator_node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR].add(
                ip_node_key
            )

            if ia_op_exec_context.operator_name == MODEL_INPUT_OP_NAME:
                self._input_ips.append(post_hook_insertion_point)

    @staticmethod
    def get_pre_hook_node_key(node_key: str, in_port_id: int = 0) -> str:
        return InsertionPointGraph.PRE_HOOK_ID_PREFIX + str(in_port_id) + " " + node_key

    @staticmethod
    def get_post_hook_node_key(node_key: str) -> str:
        return InsertionPointGraph.POST_HOOK_ID_PREFIX + node_key

    def func(self):
        node_keys_working_set = [deepcopy(node_key) for node_key in self.nodes.keys()]
        for operator_node_key in node_keys_working_set:
            # pylint: disable=unused-variable
            operator_node = self.nodes[operator_node_key]


@ignore_scope
class NNCSNetwork(nn.Module):
    def __init__(self, module: nn.Module, input_infos: List[ModelInputInfo]):
        super(NNCSNetwork, self).__init__()
        self.module = deepcopy(module)
        self._forward_signature = inspect.signature(module.forward)
        self.input_infos = input_infos

        self._wrap_inputs_fn = None
        self._extra_module_types = []
        self._user_dummy_forward_fn = None

        self._orig_context = TracingContext()
        _orig_graph_build_forward_fn = self._get_dummy_forward_fn_for_graph_building(
            with_input_tracing=True
        )
        self._graph_builder = GraphBuilder(_orig_graph_build_forward_fn)
        self._original_graph = self._graph_builder.build_graph(
            self.module, self._orig_context, as_eval=True
        )
        nx.drawing.nx_pydot.write_dot(self._original_graph._nx_graph, "graph.dot")

        self.__input_infos_based_input_wrapper = InputInfoWrapManager(
            self.input_infos, self._forward_signature, module_ref_for_device=self
        )
        self._wrap_inputs_fn = self.__input_infos_based_input_wrapper.wrap_inputs

        self._compressed_context = TracingContext()

    def forward(self, *args, **kwargs):
        with self._compressed_context as ctx:
            ctx.base_module_thread_local_replica = self
            args, kwargs = self._wrap_inputs_fn(args, kwargs)
            retval = self.module(*args, **kwargs)
        return retval

    def _get_dummy_forward_fn_for_graph_building(self, with_input_tracing):
        if self._user_dummy_forward_fn is None:
            return create_dummy_forward_fn(
                self.input_infos,
                with_input_tracing=with_input_tracing,
                wrap_inputs_fn=self._wrap_inputs_fn,
            )
        return self._user_dummy_forward_fn

    def export_onnx(
        self,
        _input,
        output_onnx,
        input_names,
        output_names,
        training=torch.onnx.TrainingMode.EVAL,
        opset_version=11,
    ):
        from packaging import version
        from nncs.fake_quantize import disable_observer

        self.apply(disable_observer)
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            from torch.onnx import OperatorExportTypes

            with torch.no_grad():
                torch.onnx.export(
                    self,
                    _input,
                    output_onnx,
                    export_params=True,
                    verbose=False,
                    training=training,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                )
        else:
            with torch.no_grad():
                torch.onnx.export(
                    self,
                    _input,
                    output_onnx,
                    export_params=True,
                    verbose=False,
                    training=training,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=opset_version,
                    do_constant_folding=True,
                )

    def merge_modules(self, keep_bn=False):
        merged_module = None
        mgraph_obj = MergedOpGraph(self._original_graph._nx_graph)
        merged_nxgraph = mgraph_obj.get_graph_with_merged_operations()
        nx.drawing.nx_pydot.write_dot(merged_nxgraph, "merged_nxgraph.dot")

        fuse_mods = []
        fuse_mods_nodenames = []
        for match in mgraph_obj.matches:
            fuse_mod = []
            for node_name in match:
                mname = get_mname_by_nname(prefix="", node_name=node_name)
                fuse_mod.append(mname)

            node_ctx = self._original_graph._nx_graph.nodes[match[0]]["op_exec_context"]
            if node_ctx.operator_name not in ["__mul__", "__add__", "__iadd__"]:
                fuse_mods_nodenames.append("\n".join(match))
                fuse_mods.append(fuse_mod)

        merged_nodenames = []
        if len(fuse_mods) != 0:
            from nncs.fuser import fuse_modules

            with training_mode_switcher(self.module, is_training=keep_bn):
                merged_module = fuse_modules.fuse_modules(self.module, fuse_mods)

            all_mod_node_names = get_all_modules(merged_module)
            mod_to_nodename = {}
            for key, value in all_mod_node_names.items():
                mod_to_nodename[value] = key

            for i in range(len(fuse_mods)):
                fuse_mod = fuse_mods[i]
                fuse_nodename = fuse_mods_nodenames[i]

                mod = get_module(merged_module, fuse_mod[0])
                merged_nodename = mod_to_nodename[mod]
                merged_node_attrs = deepcopy(merged_nxgraph.nodes[fuse_nodename])
                merged_node_attrs["mm_name"] = merged_nodename
                merged_nxgraph.add_node(fuse_nodename, **merged_node_attrs)
                merged_nodenames.append(merged_nodename)
        else:
            merged_module = deepcopy(self.module)

        return merged_nxgraph, merged_module, merged_nodenames

    def convert_to_qat_modules(
        self, merged_graph, merged_module, merged_nodenames, mapping, qconfig
    ):
        self.spec = qconfig
        qat_module = deepcopy(merged_module)

        all_mod_node_names = get_all_modules(qat_module)
        mod_to_nodename = {}
        for key, value in all_mod_node_names.items():
            mod_to_nodename[value] = key

        import nncs.dynamic_graph.patterns as p

        pattern = p.LINEAR_OPS
        matches = search_all(merged_graph, pattern)

        for match in matches:
            for node_name in match:
                if "nncs_no_quant" in node_name:
                    continue
                mname = get_mname_by_nname(prefix="", node_name=node_name)
                mod = get_module(qat_module, mname)
                node_name = mod_to_nodename[mod]
                _cls = mapping[type(mod)]
                mod.qconfig = qconfig
                qat_mod = _cls.from_float(mod)
                set_module_by_node_name(qat_module, node_name, qat_mod)

        for node_name in merged_nodenames:
            mod = get_module_by_node_name(qat_module, node_name)
            _cls = mapping[type(mod)]
            mod.qconfig = qconfig
            qat_mod = _cls.from_float(mod)
            set_module_by_node_name(qat_module, node_name, qat_mod)

        return qat_module

    def insert_activation_fq(self, merged_graph, qat_module, merged_nodenames, cs):
        ip_graph = InsertionPointGraph(merged_graph)
        # nx.drawing.nx_pydot.write_dot(ip_graph, "ip_graph.dot")

        qat_module.activation_quantizers = nn.ModuleDict()

        def fake_quantizer_hook(fq):
            def _forward_hook(self, input, output):
                return fq(output)

            return _forward_hook

        qat_module.activation_quantizers = nn.ModuleDict()
        dfs_order = nx.topological_sort(ip_graph)
        for node_key in dfs_order:
            node = ip_graph.nodes[node_key]
            node_type = node["node_type"]
            if node_type == InsertionPointGraphNodeType.INSERTION_POINT:
                predecessors = list(ip_graph.predecessors(node_key))
                assert len(predecessors) == 1
                activation_fakeq = self.spec.activation()
                predecessor = ip_graph.nodes[predecessors[0]]
                if "mm_name" in predecessor["regular_node_ref"]:
                    mm_name = predecessor["regular_node_ref"]["mm_name"]
                    if mm_name in merged_nodenames:
                        m = get_module_by_node_name(qat_module, mm_name)
                        m.register_forward_hook(fake_quantizer_hook(activation_fakeq))
                        qat_module.activation_quantizers[
                            predecessors[0]
                        ] = activation_fakeq
                else:
                    successors = list(ip_graph.successors(node_key))
                    s_no_quant_flags = False
                    for successor in successors:
                        s_nodes = ip_graph.nodes[successor]
                        if "NNCSNoQuant" in s_nodes["regular_node_ref"]["key"]:
                            s_no_quant_flags = True

                    if len(successors) == 0:
                        s_no_quant_flags = True

                    p_no_quant_flags = False
                    if "NNCSNoQuant" in predecessor["regular_node_ref"]["key"]:
                        p_no_quant_flags = True

                    if not p_no_quant_flags or (
                        p_no_quant_flags and not s_no_quant_flags
                    ):
                        ip_ia_op_exec_context = node[
                            "insertion_point_data"
                        ].ia_op_exec_context
                        if ip_ia_op_exec_context.operator_name in [
                            "__mul___RELU",
                            "__add___RELU",
                            "__mul___hardtanh",
                            "__add___hardtanh",
                            "__mul___clamp",
                            "__add___clamp",
                            "__iadd___RELU",
                        ]:
                            last_key = node_key.split("\n")[-1]
                            post_hook_key = self._original_graph._nx_graph.nodes[
                                last_key
                            ]["op_exec_context"].input_agnostic
                        else:
                            post_hook_key = ip_ia_op_exec_context

                        self._compressed_context.register_post_hooks(
                            [activation_fakeq], post_hook_key
                        )
                        qat_module.activation_quantizers[
                            predecessors[0]
                        ] = activation_fakeq
                        if cs is not None:
                            for constraint in cs.cs:
                                nodename = constraint.node_list[0]
                                if predecessors[0] == nodename:
                                    constraint_fakeq = constraint.apply_to(
                                        qat_module, merged_graph
                                    )
                                    self._compressed_context._post_hooks[
                                        post_hook_key
                                    ] = [constraint_fakeq]
                                    qat_module.activation_quantizers[
                                        predecessors[0]
                                    ] = constraint_fakeq

        self.module = qat_module

        return ip_graph

    def insert_activation_fq_within_quant_type(
        self, merged_graph, qat_module, merged_nodenames, quant_types
    ):
        ip_graph = InsertionPointGraph(merged_graph)
        nx.drawing.nx_pydot.write_dot(ip_graph, "ip_graph.dot")

        qat_module.activation_quantizers = nn.ModuleDict()

        def fake_quantizer_hook(fq):
            def _forward_hook(self, input, output):
                return fq(output)

            return _forward_hook

        qat_module.activation_quantizers = nn.ModuleDict()
        dfs_order = nx.topological_sort(ip_graph)
        for node_key in dfs_order:
            node = ip_graph.nodes[node_key]
            node_type = node["node_type"]
            if node_type == InsertionPointGraphNodeType.INSERTION_POINT:
                predecessors = list(ip_graph.predecessors(node_key))
                assert len(predecessors) == 1

                # judge predecessor in quant_types
                predecessor = ip_graph.nodes[predecessors[0]]
                pre_in_flag = False
                op_type = predecessor["regular_node_ref"][
                    "op_exec_context"
                ].input_agnostic.operator_name
                if op_type in quant_types:
                    pre_in_flag = True

                successors = list(ip_graph.successors(node_key))
                # judge successor in quant_types
                suc_in_flag = False
                for successor in successors:
                    s_node = ip_graph.nodes[successor]
                    op_type = s_node["regular_node_ref"][
                        "op_exec_context"
                    ].input_agnostic.operator_name
                    if op_type in quant_types:
                        suc_in_flag = True

                if pre_in_flag or suc_in_flag:
                    activation_fakeq = self.spec.activation()

                    if "mm_name" in predecessor["regular_node_ref"]:
                        mm_name = predecessor["regular_node_ref"]["mm_name"]
                        if mm_name in merged_nodenames:
                            m = get_module_by_node_name(qat_module, mm_name)
                            m.register_forward_hook(
                                fake_quantizer_hook(activation_fakeq)
                            )
                            qat_module.activation_quantizers[
                                predecessors[0]
                            ] = activation_fakeq
                    else:
                        ip_ia_op_exec_context = node[
                            "insertion_point_data"
                        ].ia_op_exec_context
                        self._compressed_context.register_post_hooks(
                            [activation_fakeq], ip_ia_op_exec_context
                        )
                        qat_module.activation_quantizers[
                            predecessors[0]
                        ] = activation_fakeq

        self.module = qat_module

        return ip_graph

    def apply_constriant(self, merged_graph):
        tfcs = TFConstraintSolver()
        tfcs.apply_to(merged_graph)

        cs = tfcs
        return cs

    def register_compression_module_type(
        self, compression_module_type: ExtraCompressionModuleType
    ):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type in self._extra_module_types:
            raise RuntimeError(
                "Module type {} is already registered".format(compression_module_type)
            )
        self.__setattr__(attr_name, nn.ModuleDict())
        self._extra_module_types.append(compression_module_type)

    def add_compression_module(
        self,
        module_key: str,
        module: nn.Module,
        compression_module_type: ExtraCompressionModuleType,
    ):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError(
                "Module type {} was not registered".format(compression_module_type)
            )
        self.__getattr__(attr_name)[module_key] = module

    def get_compression_modules_by_type(
        self, compression_module_type: ExtraCompressionModuleType
    ) -> nn.ModuleDict:
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError(
                "Module type {} was not registered".format(compression_module_type)
            )
        return self.__getattr__(attr_name)

    @staticmethod
    def _compression_module_type_to_attr_name(
        compression_module_type: ExtraCompressionModuleType,
    ):
        """Required for backward compatibility with checkpoints that store function and activation
        quantizers directly under corresponding attributes of NNCFNetwork."""
        if compression_module_type == ExtraCompressionModuleType.ACTIVATION_QUANTIZER:
            return "activation_quantizers"
        raise RuntimeError("Unknown extra module type")

    def quant_type(
        self, quant_spec, mapping=None, quant_prefix=None
    ):
        from nncs.quant.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS
        from nncs.quantization.tflite_constraint import UnConstraintOps, ConstraintOps

        if quant_prefix is None:
            quant_prefix = ["conv", "linear", "matmul"]

        quant_types = set()
        Ops = list(UnConstraintOps.keys()) + list(ConstraintOps.keys())

        for op in Ops:
            for qp in quant_prefix:
                if qp in op:
                    quant_types.add(op)

        quant_types = list(quant_types)

        if mapping is None:
            mapping = DEFAULT_QAT_MODULE_MAPPINGS
        else:
            pass

        merged_graph, merged_module, merged_nodenames = self.merge_modules()
        qat_modules = self.convert_to_qat_modules(
            merged_graph, merged_module, merged_nodenames, mapping, quant_spec
        )
        self.insert_activation_fq_within_quant_type(
            merged_graph, qat_modules, merged_nodenames, quant_types
        )

        return self

    def cvt_to_trt_module(self, mapping, qconfig=None):
        graph = self._original_graph._nx_graph
        all_mod_node_names = get_all_modules(self.module)
        mod_to_nodename = {}
        for key, value in all_mod_node_names.items():
            mod_to_nodename[value] = key

        # nx.drawing.nx_pydot.write_dot(self._original_graph._nx_graph, 'trt_nxgraph.dot')
        weakly_subgraphs = [
            graph.subgraph(c) for c in nx.weakly_connected_components(graph)
        ]
        for subgraph in weakly_subgraphs:
            dfs_order = nx.topological_sort(subgraph)
            for node_name in dfs_order:
                op_name = graph.nodes[node_name]["op_exec_context"].operator_name

                if (
                    op_name in ["conv2d", "linear", "max_pool2d", "adaptive_avg_pool2d"]
                    and "nncs_no_quant" not in node_name
                ):
                    mname = get_mname_by_nname(prefix="", node_name=node_name)
                    mod = get_module(self.module, mname)
                    node_name = mod_to_nodename[mod]
                    _cls = mapping[type(mod)]
                    mod.qconfig = qconfig
                    qat_mod = _cls.from_float(mod)
                    set_module_by_node_name(self.module, node_name, qat_mod)
                else:
                    pass
        return True
