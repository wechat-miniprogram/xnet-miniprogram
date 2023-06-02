import torch
import torch.nn as nn

from nncs.fuser.fuse import Fuser
from nncs.utils import _parent_name, training_mode_switcher
from nncs.fx.custom_pattern_utils import get_custom_fusion_patterns
from .custom_quantizer import ModelQuantizer


class ONNXQuantizer(ModelQuantizer):
    def __init__(self):
        super().__init__()
    
    def _fuse_fx(self, model, keep_bn=False, extra=None):
        fuser = Fuser()
        with training_mode_switcher(model, is_training=keep_bn):
            fused_mod = fuser.fuse(model, extra)
        return fused_mod

    def _convert_to_qat_modules(self, model):
        nodes = list(model.graph.nodes)

        op_nodes = []
        for node in nodes[:-1]:
            if len(node.all_input_nodes) != 0:
                op_nodes.append(node)
        
        named_modules = dict(model.named_modules())
        for op_node in op_nodes:
            if op_node.op == "call_module":
                mod = named_modules[op_node.target]
                
                type_of_mod = type(mod)
                if type_of_mod in self.mapping:
                    _cls = self.mapping[type_of_mod]
                    mod.qconfig = self.qconfig
                    qat_mod = _cls.from_float(mod)
                    parent_name, name = _parent_name(op_node.target)
                    setattr(named_modules[parent_name], name, qat_mod)

    def quant_marker(self, model):
        named_modules = dict(model.named_modules())
        nodes = list(model.graph.nodes)
        back_insert_tables = {}
        for node in nodes:
            back_insert_tables[node.name] = False
        
        for node in nodes:
            target = node.target

            if node.op in ['call_module']:
                mod = named_modules[target]
                if type(mod) in self.mapping:
                    back_insert_tables[node.name] = True
                    for arg in node.args:
                        if isinstance(arg, (torch.fx.Node)):
                            back_insert_tables[arg.name] = True
            elif node.op in['call_function']:
                assert(False), print(node.op)
            else:
                # print(node)
                pass 
        
        return back_insert_tables

    def _insert_node(self, model, fx_node, cSolver=None):
        fx_graph = model.graph
        fx_nodes = list(fx_graph.nodes)
        fx_node_name = fx_node.name
        fake_quantizer = self.qconfig.activation()
        setattr(model.activation_quantizers, fx_node_name, fake_quantizer)

        if cSolver is not None:
            pass 
            # cs = cSolver.get_constraint_by_name(fx_node_name)
            # if cs is not None:
            #     constraint_fq = cs.apply_to(model, nx_graph, quant_markers)
            #     if constraint_fq is not None:
            #         setattr(model.activation_quantizers, fx_node_name, constraint_fq)

        quantizer_name = "activation_quantizers.{}".format(fx_node_name)
        with fx_graph.inserting_after(fx_node):
            inserted_node = fx_graph.create_node(
                "call_module", quantizer_name, (fx_node,), {}
            )
            for _node in fx_nodes:
                _node.args = self._fix_succ_recursivly(
                    _node.args, fx_node, inserted_node
                )

    def _insert_activation_fq(self, model, cSolver=None):
        model.activation_quantizers = nn.ModuleDict()

        visited = {}
        nodes = list(model.graph.nodes)
        for node in nodes:
            marker = self.back_insert_tables[node.name]
            if marker and node.name not in visited:
                self._insert_node(model, node, cSolver)
        
        model.recompile()
        model.graph.lint()
        return model

    def prepare(self, model, constraint_solver,
                qconfig, mapping, **kwargs):
        self.qconfig = qconfig
        self.mapping = mapping
        additional_fusion_pattern = get_custom_fusion_patterns()
        extra = {"additional_fusion_pattern": additional_fusion_pattern}

        if "keep_bn" in kwargs:
            fused_mod = self._fuse_fx(model, keep_bn=kwargs["keep_bn"], extra=extra)
        else:
            fused_mod = self._fuse_fx(model, extra=extra)

        self.back_insert_tables = self.quant_marker(fused_mod)
        self._convert_to_qat_modules(fused_mod)

        prepared = self._insert_activation_fq(fused_mod, constraint_solver)

        return prepared