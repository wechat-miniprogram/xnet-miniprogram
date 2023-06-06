from collections import OrderedDict
import operator
import typing
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule

from nncs.fuser.fuse import Fuser
from nncs.utils import _parent_name, training_mode_switcher
from nncs.fx.constraint_solver import ConstraintSolver
from nncs.fx.custom_pattern_utils import get_custom_fusion_patterns
from nncs.fx.no_quant import FXNNCSNoQuant
from nncs.fx.quant_opset import *
from nncs.fake_quantize.fake_quantize_base import FakeQuantizeBase
from nncs.nn.intrinsic.custom_op import SharedFakeQuantize, MasterFakeQuantize


def get_node_name(module_name, prefix):
    return "{prefix}.{name}".format(prefix=prefix, name=module_name)


def get_all_modules(model, prefix=None):
    found = OrderedDict()
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(name, prefix)
        found[full_node_name] = module
        sub_found = get_all_modules(module, prefix=full_node_name)
        if sub_found:
            found.update(sub_found)
    return found


def fx_graph_to_nx_graph(model):
    nx_graph = nx.DiGraph()
    fx_graph = model.graph
    _modules = dict(model.named_modules())
    nodes = list(fx_graph.nodes)
    # TODO(ky): fix last
    for node in nodes[:-1]:
        args = node.args
        kwargs = node.kwargs
        node_info = {
            "op": node.op,
            "target": node.target,
            "meta": node.meta,
            "fx_node_name": node.name,
        }
        if len(node.all_input_nodes) == 0:
            if node.op in ["placeholder"]:
                node_info["op_type"] = "nncs_input"
                node_info["node_type"] = "quant"
            elif node.op in ["get_attr"]:
                node_info["op_type"] = "nncs_constant"
                node_info["node_type"] = "quant"
            else:
                assert False

            nx_graph.add_node(node.name, **node_info)
        else:
            if node.op == "call_module":
                op_type = type(_modules[node.target])
            elif node.op == "call_function" or node.op == "call_method":
                op_type = node.target
            else:
                assert False, "Unsupport node_op: {}".format(node.op)

            node_info["op_type"] = op_type
            node_info["node_type"] = "quant"

            if node.op == "call_function" and node.target in [getattr]:
                node_info["node_type"] = "attr"

            if node.op == "call_method" and node.target in ["size"]:
                node_info["node_type"] = "attr"

            if (
                isinstance(args[0], torch.fx.Node)
                and nx_graph.nodes[args[0].name]["node_type"] == "attr"
            ):
                node_info["node_type"] = "attr"

            nx_graph.add_node(node.name, **node_info)
            to_node_key = node.name
            edge_info = {}

            for arg in args:
                if isinstance(arg, torch.fx.node.Node):
                    from_node_key = arg.name
                    # if nx_graph.nodes[from_node_key]['node_type'] == 'quant' and \
                    #    node_info['node_type'] == 'quant':
                    #    nx_graph.add_edge(from_node_key, to_node_key, **edge_info)
                    nx_graph.add_edge(from_node_key, to_node_key, **edge_info)
                elif isinstance(
                    arg, torch.fx.immutable_collections.immutable_list
                ) or isinstance(arg, tuple):
                    for n in arg:
                        if isinstance(n, torch.fx.node.Node):
                            from_node_key = n.name
                            nx_graph.add_edge(from_node_key, to_node_key, **edge_info)
                        elif isinstance(n, (str, int, float, slice)):
                            if "attr" not in node_info:
                                node_info["attr"] = []
                            node_info["attr"].append(n)
                            nx_graph.add_node(node.name, **node_info)
                        else:
                            assert False, "Unsupported n: {}".format(n)
                elif isinstance(arg, (str, int, float)):
                    if "attr" not in node_info:
                        node_info["attr"] = []
                    node_info["attr"].append(arg)
                    nx_graph.add_node(node.name, **node_info)
                elif isinstance(arg, slice):
                    if "attr" not in node_info:
                        node_info["attr"] = []
                    node_info["attr"].append(arg)
                    nx_graph.add_node(node.name, **node_info)
                else:
                    assert False, "Unsupported node: {}".format(arg)

            for kwarg in kwargs:
                if isinstance(kwargs[kwarg], torch.fx.node.Node):
                    from_node_key = kwargs[kwarg].name
                    nx_graph.add_edge(from_node_key, to_node_key, **edge_info)
                else:
                    pass
                    # print("Debug: {}'s kwargs is {}:{}".format(node, kwarg, kwargs[kwarg]))

    attr_nodes = []
    for node in nx_graph.nodes:
        node_type = nx_graph.nodes[node]["node_type"]
        if "attr" == node_type:
            attr_nodes.append(node)
    # nx_graph.remove_nodes_from(attr_nodes)
    return nx_graph


def partition_nodes_by_nncs_marker(mod: GraphModule):
    quant_nodes, switch_nodes = [], {}
    nodes = list(mod.graph.nodes)
    named_modules = dict(mod.named_modules())

    flag = True
    for node in nodes:
        if "fx_nncs_no_quant" in node.name:
            old_flag = flag
            new_flag = not old_flag
            switch_nodes[node] = old_flag
            flag = new_flag
            continue

        if node.target in ["identity"] and type(named_modules[node.target]) in [
            nn.Identity
        ]:
            continue

        if flag:
            quant_nodes.append(node)

    return quant_nodes, switch_nodes


class ModelQuantizer(object):
    def __init__(self):
        pass

    def _fix_succ_recursivly(self, args, target_node, inserted_node):
        assert type(inserted_node) in [torch.fx.Node]
        if type(args) in [
            tuple,
            list,
            dict,
            torch.fx.immutable_collections.immutable_dict,
            torch.fx.immutable_collections.immutable_list,
        ]:
            if type(args) in [
                tuple,
                list,
                torch.fx.immutable_collections.immutable_list,
            ]:
                type_of_args = type(args)
                new_args = list(args)
                for i in range(len(new_args)):
                    new_args[i] = self._fix_succ_recursivly(
                        new_args[i], target_node, inserted_node
                    )
                args = type_of_args(new_args)
            else:
                type_of_args = type(args)
                new_args = dict(args)
                for key in new_args:
                    new_args[key] = self._fix_succ_recursivly(
                        new_args[key], target_node, inserted_node
                    )
                args = type_of_args(new_args)
        elif type(args) == type(inserted_node):
            if args == target_node:
                return inserted_node
            else:
                return args
        else:
            return args

        return args

    def erase_identity_node(self, model):
        nodes = list(model.graph.nodes)
        named_modules = dict(model.named_modules())

        for node in nodes[:-1]:
            if node.op == "call_module" and type(named_modules[node.target]) in [
                nn.Identity
            ]:
                predecessors = node.args
                assert len(predecessors) == 1
                predecessor = node.args[0]

                succs = list(node.users.keys())
                for succ in succs:
                    succ.args = self._fix_succ_recursivly(succ.args, node, predecessor)
                model.graph.erase_node(node)
                all_mods = get_all_modules(model)
                root_name = model.__class__.__name__
                pname, sname = _parent_name(node.target)
                if pname != node.target:
                    pmod = all_mods[root_name + "." + pname]
                else:
                    pmod = model
                del pmod._modules[sname]

        model.recompile()
        model.graph.lint()
        return model

    def erase_isolated_node(self, model):
        nodes = list(model.graph.nodes)

        for node in nodes:
            if len(node.all_input_nodes) == 0 and len(node.users) == 0:
                model.graph.erase_node(node)

        model.recompile()
        model.graph.lint()
        return model

    def _insert_before_node(
        self, model, nx_graph, fx_graph, fx_node, suc_fx_node, qconfig, cSolver
    ):
        fx_node_name = fx_node.name
        fake_quantizer = qconfig.activation()
        assert fx_node_name not in model.activation_quantizers
        setattr(model.activation_quantizers, fx_node_name, fake_quantizer)

        # if cSolver is not None:
        #     cs = cSolver.get_constraint_by_name(fx_node_name)
        #     if cs is not None:
        #         constraint_fq = cs.apply_to(model, nx_graph)
        #         setattr(model.activation_quantizers, fx_node_name, constraint_fq)

        quantizer_name = "activation_quantizers.{}".format(fx_node_name)
        with model.graph.inserting_before(suc_fx_node):
            inserted_node = fx_graph.create_node(
                "call_module", quantizer_name, (fx_node,), {}
            )
            type_of_args = type(suc_fx_node.args)
            args = list(suc_fx_node.args)
            idx = args.index(fx_node)
            args[idx] = inserted_node
            suc_fx_node.args = type_of_args(args)


class XNetModelQuantizer(ModelQuantizer):
    def __init__(self):
        super().__init__()

    def _fuse_fx(self, model: GraphModule, keep_bn=False, extra=None):
        fuser = Fuser()
        with training_mode_switcher(model, is_training=keep_bn):
            fused_mod = fuser.fuse(model, extra)
        return fused_mod

    def _convert_to_qat_modules(
        self, fused_mod: GraphModule, nx_graph, qconfig, mapping
    ):
        nodes = fused_mod.graph.nodes
        nodes = list(nodes)

        input_nodes = []
        op_nodes = []
        for node in nodes[:-1]:
            if len(node.all_input_nodes) == 0:
                input_nodes.append(node)
            else:
                op_nodes.append(node)

        named_modules = dict(fused_mod.named_modules())
        no_quant_nodes = []
        for op_node in op_nodes:
            if op_node.op == "call_module":
                mod = named_modules[op_node.target]
                if type(mod) in [FXNNCSNoQuant]:
                    no_quant_nodes.append(op_node.name)
                    continue

                type_of_mod = type(mod)
                if type_of_mod in mapping:
                    _cls = mapping[type_of_mod]
                    mod.qconfig = qconfig
                    qat_mod = _cls.from_float(mod)
                    parent_name, name = _parent_name(op_node.target)
                    nx_graph.nodes[op_node.name]["op_type"] = type(qat_mod)
                    setattr(named_modules[parent_name], name, qat_mod)
            else:
                pass
        return no_quant_nodes

    def quant_marker(self, model, nx_graph, no_quant_nodes, qconfig):
        fx_graph = model.graph
        fx_nodes = list(fx_graph.nodes)
        fx_node_names = [node.name for node in fx_nodes]
        model.activation_quantizers = nn.ModuleDict()

        graph = nx_graph
        weakly_subgraphs = [
            graph.subgraph(c) for c in nx.weakly_connected_components(graph)
        ]

        # record which node need to add fakequantize afterward
        results = {}
        # Create Seed Node
        for subgraph in weakly_subgraphs:
            dfs_order = nx.topological_sort(subgraph)
            for nx_node in dfs_order:
                fx_node_name_idx = fx_node_names.index(nx_node)
                fx_node = fx_nodes[fx_node_name_idx]
                quant_flag = is_quant_op(
                    fx_node, model, nx_graph, no_quant_nodes, results
                )
                results[fx_node.name] = quant_flag

        count = 0
        while True:
            number0 = 0
            for key in results:
                if results[key]:
                    number0 += 1

            for subgraph in weakly_subgraphs:
                dfs_order = nx.topological_sort(subgraph)
                for nx_node in dfs_order:
                    node_type = nx_graph.nodes[nx_node]["node_type"]
                    if node_type not in ["quant"]:
                        continue

                    fx_node_name_idx = fx_node_names.index(nx_node)
                    fx_node = fx_nodes[fx_node_name_idx]

                    op_type = nx_graph.nodes[nx_node]["op_type"]
                    if (
                        op_type in input_op_set
                        or op_type in quant_op_set
                        or op_type in not_quant_op_set
                    ):
                        pass
                    elif op_type in binary_op_set:
                        if not results[fx_node.name]:
                            assert len(fx_node.args) == 2
                            arg0 = fx_node.args[0]
                            arg1 = fx_node.args[1]
                            if isinstance(arg0, torch.fx.node.Node) and isinstance(
                                arg1, torch.fx.node.Node
                            ):
                                if results[arg0.name] and results[arg1.name]:
                                    results[fx_node.name] = True
                    elif op_type in optional_cat_quant_op_set:
                        if not results[fx_node.name]:
                            input_nodes = fx_node.all_input_nodes
                            flags = [
                                results[input_node.name] for input_node in input_nodes
                            ]
                            if False not in flags:
                                results[fx_node.name] = True
                    elif op_type in optional_relayout_quant_op_set:
                        input_nodes = fx_node.all_input_nodes
                        # assume input0 is the node operand not other attr
                        input_node = input_nodes[0]
                        if results[fx_node.name]:

                            if len(input_node.users) <= 1:
                                results[input_node.name] = True
                            else:
                                flags = []
                                for input_user in input_node.users:
                                    if input_user.op not in ["output"]:
                                        flags.append(results[input_user.name])

                                if False not in flags and input_node.type not in [
                                    typing.List[torch.Tensor]
                                ]:
                                    results[input_node.name] = True
                        else:
                            if results[input_node.name]:
                                results[fx_node.name] = True
                    elif op_type in optional_activation_quant_op_set:
                        input_nodes = fx_node.all_input_nodes
                        # assume input0 is the node operand not other attr
                        input_node = input_nodes[0]
                        if results[input_node.name]:
                            results[fx_node.name] = True
                    else:
                        print(op_type)
                        assert False
            number1 = 0
            for key in results:
                if results[key]:
                    number1 += 1

            if number1 <= number0:
                count += 1

            if count >= 2:
                break

        for fx_node_name in results:
            if fx_node_name in no_quant_nodes and nx_graph.nodes[fx_node_name][
                "op_type"
            ] not in [FXNNCSNoQuant]:
                results[fx_node_name] = False

        return results

    def _insert_node(self, model, nx_graph, fx_node, qconfig, cSolver, quant_markers):
        fx_graph = model.graph
        fx_nodes = list(fx_graph.nodes)
        fx_node_name = fx_node.name
        fake_quantizer = qconfig.activation()
        setattr(model.activation_quantizers, fx_node_name, fake_quantizer)

        if cSolver is not None:
            cs = cSolver.get_constraint_by_name(fx_node_name)
            if cs is not None:
                constraint_fq = cs.apply_to(model, nx_graph, quant_markers)
                if constraint_fq is not None:
                    setattr(model.activation_quantizers, fx_node_name, constraint_fq)

        quantizer_name = "activation_quantizers.{}".format(fx_node_name)
        with fx_graph.inserting_after(fx_node):
            inserted_node = fx_graph.create_node(
                "call_module", quantizer_name, (fx_node,), {}
            )
            for _node in fx_nodes:
                _node.args = self._fix_succ_recursivly(
                    _node.args, fx_node, inserted_node
                )

    def _insert_activation_fq(self, model, nx_graph, quant_markers, cSolver, qconfig):
        fx_graph = model.graph
        fx_nodes = list(fx_graph.nodes)
        fx_node_names = [node.name for node in fx_nodes]
        model.activation_quantizers = nn.ModuleDict()
        weakly_subgraphs = [
            nx_graph.subgraph(c) for c in nx.weakly_connected_components(nx_graph)
        ]
        visited = {}
        for subgraph in weakly_subgraphs:
            dfs_order = nx.topological_sort(subgraph)
            for node in dfs_order:
                fx_node_name_idx = fx_node_names.index(node)
                fx_node = fx_nodes[fx_node_name_idx]
                op_type = nx_graph.nodes[node]["op_type"]

                quant_marker = quant_markers[fx_node.name]
                if quant_marker and fx_node.name not in visited:
                    if op_type in onnx_ignore_op_set:
                        pass
                    elif op_type in [operator.getitem]:
                        args = fx_node.args
                        if isinstance(args[0], torch.fx.Node):
                            target = args[0].target
                            named_modules = dict(model.named_modules())
                            if target in named_modules:
                                mod = named_modules[target]
                            else:
                                mod = None
                            if isinstance(
                                mod,
                                (
                                    FakeQuantizeBase,
                                    SharedFakeQuantize,
                                    MasterFakeQuantize,
                                ),
                            ):
                                pass
                            else:
                                visited[fx_node.name] = True
                                self._insert_node(
                                    model,
                                    nx_graph,
                                    fx_node,
                                    qconfig,
                                    cSolver,
                                    quant_markers,
                                )
                        else:
                            assert False
                    else:
                        visited[fx_node.name] = True
                        self._insert_node(
                            model, nx_graph, fx_node, qconfig, cSolver, quant_markers
                        )

        model.recompile()
        model.graph.lint()
        return model

    def prepare(
        self,
        model: GraphModule,
        constraint_solver: ConstraintSolver,
        qconfig,
        mapping,
        **kwargs
    ):
        no_quant_nodes = []
        model = self.erase_identity_node(model)
        model = self.erase_isolated_node(model)
        additional_fusion_pattern = get_custom_fusion_patterns()
        extra = {"additional_fusion_pattern": additional_fusion_pattern}
        if "keep_bn" in kwargs:
            fused_mod = self._fuse_fx(model, keep_bn=kwargs["keep_bn"], extra=extra)
        else:
            fused_mod = self._fuse_fx(model, extra=extra)

        nx_graph = fx_graph_to_nx_graph(fused_mod)
        # nx.drawing.nx_pydot.write_dot(nx_graph, 'nx_graph_debug.dot')
        no_quant_mod_nodes = self._convert_to_qat_modules(
            fused_mod, nx_graph, qconfig, mapping
        )
        no_quant_nodes += no_quant_mod_nodes
        if "custom_noquant_nodes" in kwargs:
            for node in kwargs["custom_noquant_nodes"]:
                if node not in no_quant_nodes:
                    no_quant_nodes.append(node)

        quant_markers = self.quant_marker(fused_mod, nx_graph, no_quant_nodes, qconfig)
        # print(quant_markers)
        if constraint_solver is not None:
            constraint_solver.apply_to(nx_graph, no_quant_nodes, quant_markers)
        prepared = self._insert_activation_fq(
            fused_mod, nx_graph, quant_markers, constraint_solver, qconfig
        )

        return prepared


class TRTModelQuantizer(ModelQuantizer):
    def __init__(self):
        pass

    def prepare(
        self,
        model: GraphModule,
        constraint_solver: ConstraintSolver,
        qconfig,
        mapping,
        **kwargs
    ):
        model.activation_quantizers = nn.ModuleDict()
        model = self.erase_identity_node(model)
        named_modules = dict(model.named_modules())

        for node in model.graph.nodes:
            if node.op == "call_module":
                mod = named_modules[node.target]
                if type(mod) in mapping:
                    _cls = mapping[type(mod)]
                    mod.qconfig = qconfig
                    qat_mod = _cls.from_float(mod)
                    pname, sname = _parent_name(node.target)
                    setattr(named_modules[pname], sname, qat_mod)
            elif node.op == "call_function":
                if node.target in [F.adaptive_avg_pool2d]:
                    type_of_args = type(node.args)
                    this_args = list(node.args)
                    for i in range(len(node.args)):
                        _arg = node.args[i]
                        fake_quantizer = qconfig.activation()
                        setattr(
                            model.activation_quantizers,
                            node.name + "_" + str(i),
                            fake_quantizer,
                        )
                        if isinstance(_arg, torch.fx.Node):
                            with model.graph.inserting_before(node):
                                quantizer_name = "activation_quantizers.{}_{}".format(
                                    node.name, i
                                )
                                inserted_node = model.graph.create_node(
                                    "call_module", quantizer_name, (_arg,), {}
                                )
                                this_args[i] = inserted_node
                    node.args = type_of_args(this_args)
                elif node.target in [F.conv2d]:
                    input_quantizer = qconfig.activation()
                    weight_quantizer = qconfig.weight()

                    fq_names = ["input", "weight"]
                    setattr(
                        model.activation_quantizers,
                        node.name + "_" + fq_names[0],
                        input_quantizer,
                    )
                    setattr(
                        model.activation_quantizers,
                        node.name + "_" + fq_names[1],
                        weight_quantizer,
                    )

                    type_of_args = type(node.args)
                    this_args = list(node.args)
                    for i in range(2):  # input and weight
                        _arg = node.args[i]
                        if isinstance(_arg, torch.fx.Node):
                            with model.graph.inserting_before(node):
                                quantizer_name = "activation_quantizers.{}_{}".format(
                                    node.name, fq_names[i]
                                )
                                inserted_node = model.graph.create_node(
                                    "call_module", quantizer_name, (_arg,), {}
                                )
                                this_args[i] = inserted_node
                    node.args = type_of_args(this_args)
                elif node.target not in [F.hardsigmoid, operator.mul, operator.add]:
                    assert False, node.target
                else:
                    pass
            else:
                pass

        model.recompile()
        model.graph.lint()
        return model
