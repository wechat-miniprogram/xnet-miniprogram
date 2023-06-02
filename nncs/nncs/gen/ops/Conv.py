import logging

import onnx
import onnx.numpy_helper
import torch

from nncs.dynamic_graph.mod_attrs import BaseModuleAttributes
from nncs.dynamic_graph.graph import NNCSGraph

from .base import ops_registry, OpCodeGeneratorBase


@ops_registry.register(name="Conv")
class ConvOpCodeGenerator(OpCodeGeneratorBase):
    def __init__(
        self, onnx_ver=onnx.defs.onnx_opset_version(), torch_ver=torch.__version__
    ):
        super(ConvOpCodeGenerator, self).__init__(onnx_ver, torch_ver)

    def gen(self, onnx_node, value_infos, initializers, nxgraph):
        attr_value_dict = self.get_attr_value_dict(onnx_node)
        inputs_str, outputs_str = self.gen_input_output_string(
            onnx_node, initializers, self.rename_helper, self.tensor_inplace
        )

        d = len(value_infos[onnx_node.input[1]].type.tensor_type.shape.dim) - 2
        assert d in (1, 2, 3)
        nn_name = f"Conv{d}d"

        node_name = self.rename_helper.get_node_name(onnx_node.name, onnx_node.op_type)
        init_str, forward_str = [], []
        padding = 0
        if "pads" in attr_value_dict:
            padding = [attr_value_dict["pads"][i] for i in range(d)]
        else:
            logging.warning(
                "auto_pad is a DEPRECATED attribute, will not guarantee the result."
            )
            forward_str.append(
                f"{inputs_str[0]} = self.compatible_auto_pad({inputs_str[0]},
                self.{node_name}.weight.data.shape[2:], self.{node_name},
                '{attr_value_dict['auto_pad'].decode('utf-8')}')"
            )

        weights = onnx.numpy_helper.to_array(initializers[onnx_node.input[1]])

        params_str = self.gen_params_str(
            groups=attr_value_dict["group"],
            dilation=attr_value_dict.get("dilations", 1),
            out_channels=weights.shape[0],
            padding=padding,
            kernel_size=weights.shape[2:].__repr__(),
            stride=attr_value_dict.get("strides", 1),
            in_channels=weights.shape[1] * attr_value_dict["group"],
            bias=len(onnx_node.input) > 2,
        )

        init_str.append(f"self.{node_name} = nn.{nn_name}(**{{{params_str}}})")
        init_str.append(f"self.{node_name}.weight.data = {inputs_str[1]}")
        if len(onnx_node.input) > 2:
            init_str.append(f"self.{node_name}.bias.data = {inputs_str[2]}")

        forward_str.append(f"{outputs_str[0]} = self.{node_name}({inputs_str[0]})")

        node_id = self.rename_helper.get_node_id(node_name)
        node_key = node_name
        mod_attr = BaseModuleAttributes()
        mod_attr.params_str = params_str

        node_attr = {
            NNCSGraph.ID_NODE_ATTR: node_id,
            NNCSGraph.KEY_NODE_ATTR: node_key,
            NNCSGraph.OP_EXEC_CONTEXT_NODE_ATTR: None,
            NNCSGraph.MODULE_ATTRIBUTES: mod_attr,
            "op_type": nn_name,
        }

        nxgraph.add_node(node_key, **node_attr)
        nxgraph.add_edge(onnx_node.input[0], node_key)

        return {"init": init_str, "forward": forward_str}

    @staticmethod
    def gen_method():
        return """def compatible_auto_pad(self, input, kernel_spatial_shape, nn_mod, auto_pad=None, **kwargs):
        input_spatial_shape = input.shape[2:]
        d = len(input_spatial_shape)
        strides = nn_mod.stride
        dilations = nn_mod.dilation
        output_spatial_shape = [math.ceil(float(l) / float(r)) for l, r in zip(input.shape[2:], strides)]
        pt_padding = [0] * 2 * d
        pad_shape = [0] * d
        for i in range(d):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
            mean = pad_shape[i] // 2
            if auto_pad == b"SAME_UPPER":
                l, r = pad_shape[i] - mean, mean
            else:
                l, r = mean, pad_shape[i] - mean
            pt_padding.insert(0, r)
            pt_padding.insert(0, l)
        return F.pad(input, pt_padding)
        """
