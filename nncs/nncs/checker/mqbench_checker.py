import onnx
import os
import numpy as np
import networkx as nx

onnx_shared_op = {
    'GlobalAveragePool',
    'Flatten',
    'MaxPool',
    'AveragePool'
}

onnx_master_op = {
    "Concat"
}

class Checker:
    def __init__(self, onnx_path):
        self.onnx_model = onnx.load(onnx_path)
        self.xgraph = self.onnx_model.graph
        self.outputToNodeName = {}

        self.initializers = {i.name: i for i in self.onnx_model.graph.initializer}
        self.nameToNode = {i.name: i for i in self.onnx_model.graph.node}

    def replace_colon(self, name):
        return name.replace(':', '&#58')

    def onnx_graph_to_nxgraph(self):
        nxgraph = nx.DiGraph()

        inputs = self.xgraph.input
        for i in range(len(inputs)):
            node = inputs[i]
            name = node.name
            name = self.replace_colon(name)
            node_info = {
                "index_of_inputs": i
            }

            nxgraph.add_node(name, **node_info)
            self.outputToNodeName[name] = name

        nodes = self.xgraph.node
        for i in range(len(nodes)):
            node = nodes[i]
            output = node.output
            input = node.input
            name = node.name
            name = self.replace_colon(name)
            op_type = node.op_type

            node_info = {
                "input": input,
                "output": output,
                "op_type": op_type,
                "index_of_nodes": i
            }

            nxgraph.add_node(name, **node_info)

            for o in output:
                o = self.replace_colon(o)
                self.outputToNodeName[o] = name

            if len(input) != 0:
                for inp in input:
                    edge_info = {}
                    inp = self.replace_colon(inp)
                    if (inp in self.outputToNodeName and inp not in self.initializers):
                        from_node = self.outputToNodeName[inp]
                        to_node = name
                        nxgraph.add_edge(from_node, to_node, **edge_info)

        self.nxgraph = nxgraph
        # nx.drawing.nx_pydot.write_dot(nxgraph, 'nxgraph_debug.dot')

    def parse_scale_zp(self, node):
        scale_name = node['input'][1]
        try:
            zp_name = node['input'][2]
        except:
            import ipdb; ipdb.set_trace()
            asd = 0

        scaleNodeName = self.replace_colon(scale_name)
        if scaleNodeName in self.outputToNodeName:
            node_name = self.outputToNodeName[scaleNodeName]
            snode = self.nameToNode[node_name]
            svalue = np.frombuffer(snode.attribute[0].t.raw_data, dtype=np.float32)
        elif scale_name in self.initializers:
            scale_initer = self.initializers[scale_name]
            data_type = scale_initer.data_type
            assert(data_type == 1)
            if len(scale_initer.float_data) != 0:
                svalue = np.array(scale_initer.float_data).astype(np.float32)
            elif len(scale_initer.raw_data) != 0:
                svalue = np.frombuffer(scale_initer.raw_data, dtype=np.float32)
            else:
                assert(False)
            
        else:
            assert(False)

        zpNodeName = self.replace_colon(zp_name)
        if zpNodeName in self.outputToNodeName:
            node_name = self.outputToNodeName[zpNodeName]
            znode = self.nameToNode[node_name]
            zvalue = np.frombuffer(znode.attribute[0].t.raw_data, dtype=np.int8)
        elif zp_name in self.initializers:
            zp_initer = self.initializers[zp_name]
            data_type = zp_initer.data_type
            assert(data_type in [1, 2, 3])
            if data_type == 1:
                if len(zp_initer.float_data) != 0:
                    zvalue = np.array(zp_initer.float_data).astype(np.float32)
                elif len(zp_initer.raw_data) != 0:
                    zvalue = np.frombuffer(zp_initer.raw_data, dtype=np.float32)
                else:
                    assert(False)
            elif data_type == 2:
                import ipdb; ipdb.set_trace()
                zvalue = np.frombuffer(zp_initer.raw_data, dtype=np.int8)
                zvalue = zvalue.astype(np.float32)
            elif data_type == 3:
                import ipdb; ipdb.set_trace()
                zvalue = np.array(zp_initer.int32_data).astype(np.float32)
            else:
                assert(False)
        else:
            assert(False)

        return svalue, zvalue

    def check_tflite_constraints(self):
        nxgraph = self.nxgraph
        weakly_subgraphs = [
            nxgraph.subgraph(c) for c in nx.weakly_connected_components(nxgraph)
        ]
        for subgraph in weakly_subgraphs:
            dfs_order = nx.topological_sort(subgraph)
            for node in dfs_order:

                nxnode = nxgraph.nodes[node]
                if 'op_type' not in nxnode:
                    continue

                if nxnode['op_type'] in onnx_shared_op:
                    # print("processing shared: {}".format(node))
                    predecessors = list(nxgraph.predecessors(node))
                    successors = list(nxgraph.successors(node))

                    predecessor = nxgraph.nodes[predecessors[0]]
                    successor = nxgraph.nodes[successors[0]]
                    
                    if successor['op_type'] in ['LearnablePerTensorAffine'] and \
                        predecessor['op_type'] in ['LearnablePerTensorAffine']:
                        scale0, zp0 = self.parse_scale_zp(successor)

                        scale1, zp1 = self.parse_scale_zp(predecessor)

                        sflag = np.allclose(scale0, scale1, rtol=1e-05, atol=1e-08)
                        zflag = np.allclose(zp0, zp1, rtol=1e-05, atol=1e-08)

                        if not sflag or not zflag:
                            return False
                elif nxnode['op_type'] in onnx_master_op:
                    # print("processing master: {}".format(node))
                    predecessors = list(nxgraph.predecessors(node))
                    successors = list(nxgraph.successors(node))
                    s = nxgraph.nodes[successors[0]]

                    if s['op_type'] in ['LearnablePerTensorAffine']:
                        sscale, szp = self.parse_scale_zp(s)
                        for predecessor in predecessors:
                            p = nxgraph.nodes[predecessor]
                            if p['op_type'] in ['LearnablePerTensorAffine']:
                                pscale, pzp = self.parse_scale_zp(p)
                                sflag = np.allclose(sscale, pscale, rtol=1e-05, atol=1e-08)
                                zflag = np.allclose(szp, pzp, rtol=1e-05, atol=1e-08)
                                if not sflag or not zflag:
                                    return False
                else:
                    pass

        return True


models = [
    "alexnet", "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50", "efficientnet_b0", "fcn_resnet50", "googlenet", "inception_v3",
    "lraspp_mobilenet_v3_large", "mnasnet0_5", "mobilenet_v2", "regnet_x_1_6gf",
    "resnet18", "resnext50_32x4d", "shufflenet_v2_x1_0", "squeezenet1_0",
    "vgg19_bn", "wide_resnet50_2"
]

# nncs_base_path = "/Users/kangyang/pytorch_quantization/nncs_quantize_onnxs"
# ppq_base_path = "/Users/kangyang/pytorch_quantization/ppq_quantize_onnxs"
# pytorch_base_path = "/Users/kangyang/pytorch_quantization/pytorch_quantize_sim_onnxs"
mqbench_base_path = "/Users/kangyang/pytorch_quantization/mqbench_quantize_onnxs"

for model in models:
    checker = Checker(os.path.join(mqbench_base_path, model + ".onnx.onnx"))
    checker.onnx_graph_to_nxgraph()
    flag = checker.check_tflite_constraints()
    print(model, flag)


