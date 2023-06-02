from typing import Any, Callable, Dict
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.graph import Node

from torch.quantization.fx.fusion_patterns import (
    FuseHandler,
)
from torch.quantization.fx.quantization_types import QuantizerCls

from nncs.utils import _parent_name
from nncs.fx.custom_pattern_utils import register_fusion_pattern
from nncs.fuser.fuser_method_mappings import get_fuser_method
from nncs.nn.intrinsic.custom_op.learnable_relu import _LearnableClip


@register_fusion_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Linear))
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm1d, torch.nn.Linear))
)
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm1d, torch.nn.Linear)))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Linear))
class LinearBNReLUFusion(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        self.relu_node = None
        self.bn_node = None
        if (
            node.op == "call_function" and node.target in [torch.nn.functional.relu]
        ) or (
            node.op == "call_module"
            and type(quantizer.modules[node.target]) in [torch.nn.ReLU]
        ):
            self.relu_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm1d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.linear_node = node
        self.linear = quantizer.modules[self.linear_node.target]

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:

        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get(
            "additional_fuser_method_mapping", {}
        )
        op_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to create a relu module for each match
            if self.relu_node.op == "call_module":
                type_of_act = type(quantizer.modules[self.relu_node.target])
                if type_of_act in [torch.nn.ReLU, torch.nn.ReLU6]:
                    relu = type_of_act(quantizer.modules[self.relu_node.target].inplace)
                else:
                    relu = type_of_act()
            else:
                # TODO: get inplace argument from functional
                relu = torch.nn.ReLU()
            op_list.append(relu)
            relu.training = self.linear.training
            if self.bn_node is not None:
                op_list.append(self.bn)
            op_list.append(self.linear)
        else:
            assert self.bn_node is not None
            op_list.append(self.bn)
            op_list.append(self.linear)

        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        linear_parent_name, linear_name = _parent_name(self.linear_node.target)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)
        setattr(quantizer.modules[linear_parent_name], linear_name, fused)

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        # relu may be used multiple times, so we don't set relu to identity
        return quantizer.fused_graph.node_copy(self.linear_node, load_arg)
        pass


@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.BatchNorm3d, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm1d, torch.nn.Conv1d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((torch.nn.ReLU, (torch.nn.BatchNorm3d, torch.nn.Conv3d)))
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm1d, torch.nn.Conv1d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm3d, torch.nn.Conv3d))
)
class ConvBNReLUFusion(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.relu_node = None
        self.bn_node = None
        if (node.op == "call_function" and node.target is torch.nn.functional.relu) or (
            node.op == "call_module"
            and type(quantizer.modules[node.target]) == torch.nn.ReLU
        ):
            self.relu_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        if type(quantizer.modules[node.target]) in [
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
        ]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get(
            "additional_fuser_method_mapping", {}
        )
        op_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to create a relu module for each match
            if self.relu_node.op == "call_module":
                relu = torch.nn.ReLU(quantizer.modules[self.relu_node.target].inplace)
            else:
                # TODO: get inplace argument from functional
                relu = torch.nn.ReLU()
            op_list.append(relu)
            relu.training = self.conv.training
            if self.bn_node is not None:
                op_list.append(self.bn)
            op_list.append(self.conv)
        else:
            assert self.bn_node is not None
            op_list.append(self.bn)
            op_list.append(self.conv)

        # the modules are added in order of relu - bn - conv
        # so we need to correct it
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)
        setattr(quantizer.modules[conv_parent_name], conv_name, fused)

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        # relu may be used multiple times, so we don't set relu to identity
        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)


@register_fusion_pattern((torch.nn.functional.relu6, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU6, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU6, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern(
    (torch.nn.functional.relu6, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
)
class ConvBNClipFunction(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.act_node = None
        self.bn_node = None
        if (
            node.op == "call_function"
            and node.target in [torch.nn.functional.relu6]
            or node.op == "call_module"
            and type(quantizer.modules[node.target]) in [torch.nn.ReLU6]
        ):
            self.act_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        if type(quantizer.modules[node.target]) in [
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
        ]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get(
            "additional_fuser_method_mapping", {}
        )
        op_list = []
        if self.act_node is not None:
            if self.act_node.op == "call_module":
                relu = torch.nn.ReLU6(quantizer.modules[self.act_node.target].inplace)
            else:
                # TODO: get inplace argument from functional
                relu = torch.nn.ReLU6()
            op_list.append(relu)
            relu.training = self.conv.training

        if self.bn_node is not None:
            op_list.append(self.bn)
        op_list.append(self.conv)
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)

        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        setattr(quantizer.modules[conv_parent_name], conv_name, fused)

        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())

        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)


@register_fusion_pattern((_LearnableClip, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
@register_fusion_pattern((_LearnableClip, torch.nn.Conv2d))
class ConvBNLClipFunction(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.act_node = None
        self.bn_node = None
        if (
            node.op == "call_function"
            and node.target in [_LearnableClip]
            or node.op == "call_module"
            and type(quantizer.modules[node.target]) in [_LearnableClip]
        ):
            self.act_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        if type(quantizer.modules[node.target]) in [
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
        ]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get(
            "additional_fuser_method_mapping", {}
        )
        op_list = []
        if self.act_node is not None:
            if self.act_node.op == "call_module":
                act = quantizer.modules[self.act_node.target]
            else:
                assert False, "_LearnableClip not call_function"
            op_list.append(act)
            act.training = self.conv.training

        if self.bn_node is not None:
            op_list.append(self.bn)
        op_list.append(self.conv)
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)

        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        setattr(quantizer.modules[conv_parent_name], conv_name, fused)

        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())

        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)


@register_fusion_pattern((torch.nn.ReLU6, torch.nn.ConvTranspose2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.ConvTranspose2d))
@register_fusion_pattern((torch.nn.functional.relu6, torch.nn.ConvTranspose2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.ConvTranspose2d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d))
@register_fusion_pattern(
    (torch.nn.ReLU6, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d))
)
@register_fusion_pattern(
    (torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu6, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d))
)
class ConvTransposedBNReLUFusion(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.relu_node = None
        self.bn_node = None
        if (
            node.op == "call_function"
            and node.target in [torch.nn.functional.relu, torch.nn.functional.relu6]
        ) or (
            node.op == "call_module"
            and type(quantizer.modules[node.target]) in [torch.nn.ReLU, torch.nn.ReLU6]
        ):
            self.relu_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm2d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get(
            "additional_fuser_method_mapping", {}
        )
        op_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to create a relu module for each match
            if self.relu_node.op == "call_module":
                if type(quantizer.modules[self.relu_node.target]) == torch.nn.ReLU:
                    relu = torch.nn.ReLU(
                        quantizer.modules[self.relu_node.target].inplace
                    )
                elif type(quantizer.modules[self.relu_node.target]) == torch.nn.ReLU6:
                    relu = torch.nn.ReLU6(
                        quantizer.modules[self.relu_node.target].inplace
                    )
                else:
                    assert False, type(quantizer.modules[self.relu_node.target])
            elif self.relu_node.op == "call_function":
                if self.relu_node.target == torch.nn.functional.relu:
                    # TODO: get inplace argument from functional
                    relu = torch.nn.ReLU()
                elif self.relu_node.target == torch.nn.functional.relu6:
                    relu = torch.nn.ReLU6()
                else:
                    assert False, self.relu_node.target
            else:
                assert False, self.relu_node.op

            op_list.append(relu)
            relu.training = self.conv.training
            if self.bn_node is not None:
                op_list.append(self.bn)
            op_list.append(self.conv)
        else:
            assert self.bn_node is not None
            op_list.append(self.bn)
            op_list.append(self.conv)

        # the modules are added in order of relu - bn - conv
        # so we need to correct it
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)
        setattr(quantizer.modules[conv_parent_name], conv_name, fused)

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        # relu may be used multiple times, so we don't set relu to identity
        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)


class BinaryClip(nn.Module):
    def __init__(self, binary_op, clip):
        super(BinaryClip, self).__init__()
        if binary_op == operator.add:
            self.binary_op = torch.add
        elif binary_op == operator.mul:
            self.binary_op = torch.mul
        else:
            assert False

        self.clip = clip

    def forward(self, a, b):
        binary_out = self.binary_op(a, b)
        o = self.clip(binary_out)
        return o


functional_mapping = {
    torch.nn.functional.relu6: torch.nn.ReLU6,
    torch.nn.functional.relu: torch.nn.ReLU,
}


@register_fusion_pattern((_LearnableClip, operator.mul))
@register_fusion_pattern((torch.nn.ReLU6, operator.mul))
@register_fusion_pattern((torch.nn.functional.relu6, operator.mul))
@register_fusion_pattern((torch.nn.ReLU, operator.mul))
@register_fusion_pattern((torch.nn.functional.relu, operator.mul))
@register_fusion_pattern((_LearnableClip, operator.add))
@register_fusion_pattern((torch.nn.ReLU6, operator.add))
@register_fusion_pattern((torch.nn.functional.relu6, operator.add))
@register_fusion_pattern((torch.nn.ReLU, operator.add))
@register_fusion_pattern((torch.nn.functional.relu, operator.add))
class BinaryClipFunction(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.act_node = None
        if (
            node.op == "call_function"
            and node.target in [torch.nn.functional.relu6, torch.nn.functional.relu]
            or node.op == "call_module"
            and type(quantizer.modules[node.target])
            in [torch.nn.ReLU, torch.nn.ReLU6, _LearnableClip]
        ):
            self.act_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]

        self.binary_node = node

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        if self.act_node is not None:
            if self.act_node.op == "call_module":
                type_of_act = type(quantizer.modules[self.act_node.target])
                if torch.nn.ReLU6 == type_of_act:
                    clip = torch.nn.ReLU6(
                        quantizer.modules[self.act_node.target].inplace
                    )
                elif torch.nn.ReLU == type_of_act:
                    clip = torch.nn.ReLU(
                        quantizer.modules[self.act_node.target].inplace
                    )
                elif _LearnableClip == type_of_act:
                    clip = quantizer.modules[self.act_node.target]
                else:
                    assert False
            else:
                clip = functional_mapping[type(self.act_node.target)]()

        fused = BinaryClip(self.binary_node.target, clip)
        name = (
            self.binary_node.target.__name__.replace(".", "_")
            + "_"
            + self.act_node.target.replace(".", "_")
        )

        setattr(quantizer.root_model, name, fused)

        # self.binary_node.name = name
        self.binary_node.target = name
        self.binary_node.op = "call_module"
        return quantizer.fused_graph.node_copy(self.binary_node, load_arg)


class SharedOpDropout(nn.Module):
    def __init__(self, anyOp, dropout):
        super(SharedOpDropout, self).__init__()
        self.anyOp = anyOp
        self.dropout = dropout

    def forward(self, *args, **kwargs):
        A = self.anyOp(*args, **kwargs)
        out = self.dropout(A)
        return out


@register_fusion_pattern((nn.Dropout, torch.flatten))
class SharedOpDropoutFunction(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.dropout_node = None
        if (
            node.op == "call_function"
            and node.target in [F.dropout]
            or node.op == "call_module"
            and type(quantizer.modules[node.target]) in [nn.Dropout]
        ):
            self.dropout_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]

        self.any_node = node

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        if self.dropout_node is not None:
            if self.dropout_node.op == "call_module":
                mod = quantizer.modules[self.dropout_node.target]
                type_of_act = type(mod)
                dropout = type_of_act(p=mod.p, inplace=mod.inplace)
                dropout_name = self.dropout_node.target.replace(".", "_")
            else:
                assert False

        if self.any_node.op == "call_method" or "call_function":
            fused = SharedOpDropout(self.any_node.target, dropout)
            any_name = self.any_node.target.__name__.replace(".", "_")

        elif self.any_node.op == "call_module":
            mod = quantizer.modules[self.any_node.target]
            fused = SharedOpDropout(mod, dropout)
            any_name = self.any_node.target.replace(".", "_")
        else:
            assert False

        name = any_name + "_" + dropout_name

        setattr(quantizer.root_model, name, fused)

        # self.binary_node.name = name
        self.any_node.target = name
        self.any_node.op = "call_module"
        return quantizer.fused_graph.node_copy(self.any_node, load_arg)


class RelayoutContiguousOp(nn.Module):
    def __init__(self, target_func):
        super(RelayoutContiguousOp, self).__init__()
        self.target_func = target_func

    def forward(self, *args, **kwargs):
        out = getattr(args[0], self.target_func)(*args[1:], **kwargs)
        out = out.contiguous()

        return out


class RelayoutFunctionContiguousOp(nn.Module):
    def __init__(self, target_func):
        super(RelayoutFunctionContiguousOp, self).__init__()
        self.target_func = target_func

    def forward(self, *args, **kwargs):
        out = self.target_func(*args, **kwargs)
        out = out.contiguous()

        return out


@register_fusion_pattern(("contiguous", torch.transpose))
@register_fusion_pattern(("contiguous", "transpose"))
class RelayoutContiguousFunction(FuseHandler):
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        self.contiguous_node = None
        if node.op == "call_method" and node.target in ["contiguous"]:
            self.contiguous_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]

        self.relayout_node = node

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        if self.relayout_node.op == "call_method":
            fused = RelayoutContiguousOp(self.relayout_node.target)
            relayout_node_name = self.relayout_node.target.replace(".", "_")
        elif self.relayout_node.op == "call_function":
            fused = RelayoutFunctionContiguousOp(self.relayout_node.target)
            relayout_node_name = self.relayout_node.target.__name__.replace(".", "_")
        else:
            assert False

        contiguous_node_name = self.contiguous_node.target.replace(".", "_")
        name = relayout_node_name + "_" + contiguous_node_name
        setattr(quantizer.root_model, name, fused)
        self.relayout_node.target = name
        self.relayout_node.op = "call_module"
        return quantizer.fused_graph.node_copy(self.relayout_node, load_arg)
