import os
import re
import shutil
import logging
from collections import Counter, OrderedDict

import numpy as np
import networkx as nx
import onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array
from nncs.gen.ops.base import ops_registry
from nncs.gen.onnx_gen_template import CodeGenTemplate
from nncs.gen.ops import *


class RenameHelper:
    def __init__(self, simplify_names=False):
        self.simplify_names = simplify_names

        self.tensor_name_mapping = {}
        self.tensor_name_counter = Counter()
        self.node_name_mapping = {}
        self.node_name_counter = Counter()

        self.tensor_counter = 0
        self.node_counter = Counter()

        self.node_key_mapping = OrderedDict()

    def get_tensor_name(self, tensor_name):
        if self.simplify_names:
            return self.get_simplify_tensor_name(tensor_name)
        if tensor_name.isnumeric():
            self.tensor_name_mapping[tensor_name] = f"t_{tensor_name}"
            return f"t_{tensor_name}"
        return tensor_name

    def get_simplify_tensor_name(self, tensor_name):
        if tensor_name in self.tensor_name_mapping:
            return self.tensor_name_mapping[tensor_name]
        suffix = self.tensor_counter
        self.tensor_counter += 1
        sim_tensor_name = f"t_{suffix}"
        self.tensor_name_mapping[tensor_name] = sim_tensor_name
        return self.tensor_name_mapping[tensor_name]

    def get_node_name(self, node_name, op_type):
        if self.simplify_names or not node_name:
            return self.get_simplify_node_name(node_name, op_type)
        return f"n_{node_name}"

    def get_simplify_node_name(self, node_name, op_type):
        idx = self.node_counter[op_type]
        self.node_counter[op_type] += 1
        self.node_name_mapping[node_name] = f"n_{op_type}_{idx}"
        return self.node_name_mapping[node_name]

    def get_node_id(self, node_key):
        if node_key not in self.node_key_mapping:
            self.node_key_mapping[node_key] = len(self.node_key_mapping)

        return self.node_key_mapping[node_key]


class OnnxCodeGenerator:
    def __init__(
        self,
        onnx_model=None,
        output_dir=None,
        simplify_names=False,
        tensor_inplace=False,
        continue_on_error=False,
        embedding_conf=None,
        shape_infer=True,
    ):
        self.onnx_model = onnx_model
        self.output_dir = output_dir
        self.rename_helper = RenameHelper(simplify_names)
        self.tensor_inplace = tensor_inplace
        self.continue_on_error = continue_on_error
        self.embedding_conf = embedding_conf
        self.shape_infer = shape_infer
        self.init_parts = []
        self.forward_parts = []
        self.method_parts = {}

        self._nx_graph = nx.DiGraph()

    def add_init_part(self, m):
        if type(m) in (list, tuple, set):
            self.init_parts.extend(m)
        else:
            self.init_parts.append(m)

    def add_forward_part(self, m):
        if type(m) in (list, tuple, set):
            self.forward_parts.extend(m)
        else:
            self.forward_parts.append(m)

    def add_forward_return(self, outputs_value_infos):
        return_list = [
            f"{self.rename_helper.get_tensor_name(o.name)}" for o in outputs_value_infos
        ]
        self.forward_parts.append(f"return {', '.join(return_list)}")

    def vis_graph(self):
        nx.drawing.nx_pydot.write_dot(self._nx_graph, "graph.dot")

    def preprocess_onnx_model(self):
        for n in self.onnx_model.graph.node:
            inputs, outputs = [], []
            for ls, ns in ((inputs, n.input), (outputs, n.output)):
                for k in ns:
                    new_i = re.sub("[:/.]", "_", k)
                    ls.append(new_i)
                    if k != new_i:
                        logging.info(f"Input name {k} is changed to {new_i}.")
                    self.rename_helper.tensor_name_counter[new_i] += 1

            n.ClearField("input")
            n.input.extend(inputs)
            n.ClearField("output")
            n.output.extend(outputs)

            old_name = n.name
            n.name = re.sub("[:/.]", "_", n.name)
            if old_name != n.name:
                logging.info(f"Node name {old_name} is changed to {n.name}.")
            self.rename_helper.node_name_counter[n.name] += 1

        for ns in (
            self.onnx_model.graph.input,
            self.onnx_model.graph.output,
            self.onnx_model.graph.initializer,
        ):
            for k in ns:
                old_name = k.name
                k.name = re.sub("[:/.]", "_", k.name)
                if old_name != k.name:
                    logging.info(f"Tensor name {old_name} is changed to {k.name}.")
                self.rename_helper.tensor_name_counter[k.name] += 1

        for f in (self.onnx_model.graph.value_info,):
            for i in f:
                old_name = i.name
                i.name = re.sub("[:/.]", "_", i.name)
                if old_name != i.name:
                    logging.info(f"Tensor name {old_name} is changed to {i.name}.")
                self.rename_helper.tensor_name_counter[i.name] += 1
        onnx.save(self.onnx_model, os.path.join(self.output_dir, "tmp_prepro.onnx"))

    def add_forward_input(self, input_value_infos, initializers):
        initializer_names = initializers.keys()
        inputs_value_names = input_value_infos.keys()
        return_list = [
            f"{self.rename_helper.get_tensor_name(i)}"
            for i in (inputs_value_names - initializer_names)
        ]
        if len(return_list) == 1:
            self.forward_parts.append(f"{return_list[0]}, = inputs")
        else:
            self.forward_parts.append(f"{', '.join(return_list)} = inputs")

        for input_name in return_list:
            node_attrs = {}
            node_attrs["proto"] = input_value_infos[input_name]
            self._nx_graph.add_node(input_name, **node_attrs)

        return return_list

    def gen_test_run_model_code(self):
        numpy_input_str = []
        initializer_names = {i.name for i in self.onnx_model.graph.initializer}
        for i in self.onnx_model.graph.input:
            if i.name in initializer_names:
                continue
            dtype = TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type]
            shape = []
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param != "":
                    shape.append(1)
                else:
                    shape.append(d.dim_value)
            if shape:
                numpy_input_str.append(
                    f"torch.from_numpy(np.random.randn(*{[s if s > 1
                    else 1 for s in shape].__repr__()}).astype(np.{dtype.name}))"
                )
            else:
                numpy_input_str.append(
                    f"torch.from_numpy(np.random.randn(1).astype(np.{dtype.name}))"
                )
        test_run_model = [
            f"""@torch.no_grad()
def test_run_model(inputs=[{', '.join(numpy_input_str)}]):""",
            "model = Model()",
            "model.eval()",
        ]
        test_run_model.extend(["rs = model(*inputs)", "print(rs)", "return rs"])
        return """
    """.join(
            test_run_model
        )

    def gen_model_code(self):
        return CodeGenTemplate.model(
            model_init="""
        """.join(
                self.init_parts
            ),
            model_forward="""
        """.join(
                self.forward_parts
            ),
            model_method="""
        """.join(
                self.method_parts.values()
            ),
            test_run_model=self.gen_test_run_model_code(),
        )

    def add_attr_to_op_code_generator(self, op_code_gen):
        for k, v in {
            "rename_helper": self.rename_helper,
            "tensor_inplace": self.tensor_inplace,
            "embedding_conf": self.embedding_conf,
        }.items():
            if hasattr(op_code_gen, k):
                setattr(op_code_gen, k, v)

    def run(self):
        self.preprocess_onnx_model()
        initializers = {i.name: i for i in self.onnx_model.graph.initializer}
        input_value_infos = {i.name: i for i in self.onnx_model.graph.input}
        output_value_infos = {i.name: i for i in self.onnx_model.graph.output}

        value_infos = {}
        value_infos.update(input_value_infos)
        value_infos.update(output_value_infos)
        value_infos.update({i.name: i for i in self.onnx_model.graph.value_info})

        self.input_names = self.add_forward_input(input_value_infos, initializers)

        for n in self.onnx_model.graph.node:
            op_type = n.op_type

            if op_type not in ops_registry.registry_dict:
                assert False, f"unregister {op_type}"
            op_code_gen = ops_registry.registry_dict[op_type]()
            self.add_attr_to_op_code_generator(op_code_gen)

            if (
                hasattr(op_code_gen, "gen_method")
                and n.op_type not in self.method_parts
            ):
                self.method_parts[n.op_type] = op_code_gen.gen_method()

            gened = op_code_gen.gen(n, value_infos, initializers, self._nx_graph)
            self.add_init_part(gened["init"])
            self.add_forward_part(gened["forward"])

        self.add_forward_return(self.onnx_model.graph.output)
        gened_code = self.gen_model_code()
        with open(os.path.join(self.output_dir, "model.py"), "w") as f:
            f.write(gened_code)

        shutil.rmtree(os.path.join(self.output_dir, "variables"), ignore_errors=True)
        os.makedirs(os.path.join(self.output_dir, "variables"))
        import ipdb

        ipdb.set_trace()
        for k, v in initializers.items():
            np.save(
                os.path.join(
                    self.output_dir,
                    "variables",
                    f"{self.rename_helper.get_tensor_name(k)}.npy",
                ),
                to_array(v),
            )


def get_onnx_code_generator(
    onnx_model,
    output_dir,
    tensor_inplace=False,
    simplify_names=False,
    continue_on_error=False,
    embedding_conf_file=None,
    shape_infer=False,
):
    kwargs = {
        "output_dir": output_dir,
        "simplify_names": simplify_names,
        "tensor_inplace": tensor_inplace,
        "continue_on_error": continue_on_error,
        "shape_infer": shape_infer,
    }
    if type(onnx_model) != onnx.ModelProto:
        assert os.path.exists(onnx_model), f"ONNX model {onnx_model} does not exist."
        assert os.path.isfile(onnx_model), f"{onnx_model} is not a file"
        kwargs["onnx_model"] = onnx.load(onnx_model)
    else:
        kwargs["onnx_model"] = onnx_model

    if os.path.exists(output_dir):
        # assert(False), f"output_dir {output_dir} is exists."
        pass
    else:
        os.makedirs(output_dir)

    kwargs["embedding_conf"] = embedding_conf_file

    return OnnxCodeGenerator(**kwargs)
