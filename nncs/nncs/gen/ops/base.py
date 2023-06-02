import logging

import torch
import onnx

from nncs.common.utils.registry import Registry


class OpCodeGeneratorBase:
    def __init__(
        self, onnx_ver=onnx.defs.onnx_opset_version(), torch_ver=torch.__version__
    ):
        self.onnx_ver = onnx_ver
        self.torch_ver = torch_ver
        self.onnx_op = self.__class__.__name__.replace("OpCodeGenerator", "")
        self.schema = onnx.defs.get_schema(self.onnx_op, max_inclusive_version=onnx_ver)

        self.rename_helper = None
        self.tensor_inplace = None

        if self.schema is not None:
            self.attr_default = {}
            for a, i in self.schema.attributes.items():
                try:
                    default_value = onnx.helper.get_attribute_value(i.default_value)
                    self.attr_default[a] = default_value
                except Exception:  # pylint: disable=broad-except
                    logging.warning(
                        f"Cannot get default value for {a} of {self.onnx_op}."
                    )

    def gen(self, onnx_node, value_infos, initializers, nxgraph):
        raise Exception

    def get_attr_value_dict(self, node):
        attr_value_dict = {}
        for a in node.attribute:
            attr_value_dict[a.name] = onnx.helper.get_attribute_value(a)
        attr_value_dict = dict(
            list(self.attr_default.items()) + list(attr_value_dict.items())
        )
        return attr_value_dict

    def gen_input_output_string(
        self,
        node,
        initializers,
        rename_helper,
        tensor_inplace=False,
        input_num=None,
        output_num=None,
    ):
        inputs_str, outputs_str = [], []
        input_num, output_num = input_num or len(node.input), output_num or len(
            node.output
        )
        for idx, (num, f, ls) in enumerate(
            (
                (input_num, node.input, inputs_str),
                (output_num, node.output, outputs_str),
            )
        ):
            for i in range(num):
                if (
                    idx == 1
                    and i == 0
                    and tensor_inplace
                    and len(node.input) > 0
                    and node.input[0] not in initializers
                    and rename_helper.tensor_name_counter[f[i]] == 2
                    and rename_helper.tensor_name_counter[node.input[0]] == 2
                ):
                    tensor_name = node.input[0]
                    rename_helper.tensor_name_mapping[
                        f[i]
                    ] = rename_helper.get_tensor_name(tensor_name)
                else:
                    tensor_name = f[i]
                formatter = "{}"
                if tensor_name in initializers:
                    formatter = 'self._vars["{}"]'
                s = formatter.format(rename_helper.get_tensor_name(tensor_name))
                ls.append(s)
        return inputs_str, outputs_str

    def gen_params_str(self, **kwargs):
        params = []
        for k, v in kwargs.items():
            v_str = v if type(v) == str else v.__repr__()
            params.append(f"'{k}': {v_str}")
        return ", ".join(params).__repr__()[1:-1]

    def check_in_init(self, targets, initializers):
        lacks = []
        rs = [None] * len(targets)
        for i, (_, n) in enumerate(targets):
            init = initializers.get(n, None)
            if init is None:
                lacks.append(n)
            rs[i] = init
        if lacks:
            raise Exception(
                f"Currently {self.__class__} only support all of {lacks.__repr__()} is in initializers."
            )
        return rs

    def get_shape(self, value, value_infos):
        if value not in value_infos:
            return None
        shape = []
        for d in value_infos[value].type.tensor_type.shape.dim:
            if d.dim_param != "":
                shape.append(-1)
            else:
                shape.append(d.dim_value)
        return shape


ops_registry = Registry("op_code_gen")
