import copy
import torch
import numpy as np


def verify_o2t_convert(onnx_model_path, _graph_model = None, input_shapes=None):
    import onnxruntime as ort
    from onnx2torch import convert
    if _graph_model is not None:
        graph_model = copy.deepcopy(_graph_model)
    else:
        graph_model = convert(onnx_model_path)

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    ort_sess = ort.InferenceSession(onnx_model_path, sess_options=options)
    inputs = ort_sess.get_inputs()
    input_shapes = [inp.shape for inp in inputs]

    for inp_shape in input_shapes:
        if len(inp_shape) > 1:
            for inp_s in inp_shape[1:]:
                if not isinstance(inp_s, int):
                    assert(False), "input_shapes have dynamic shape"

    sess_inputs = dict()
    pth_inputs = []
    for inp, inp_shape in zip(inputs, input_shapes):
        if isinstance(inp_shape[0], str):
            inp_shape[0] = 1
        if inp.type in ['tensor(int32)']:
            inp_tensor = torch.randint(0, 255, inp_shape).int()
        else:
            inp_tensor = torch.rand(*inp_shape)

        sess_inputs[inp.name] = inp_tensor.numpy()
        pth_inputs.append(inp_tensor)

    ort_outputs = ort_sess.run(None, sess_inputs)

    graph_model.eval()
    pth_outputs = graph_model(*pth_inputs)

    if not isinstance(pth_outputs, (tuple, list)):
        pth_outputs = [pth_outputs]

    results = []
    for ort_output, pth_output in zip(ort_outputs, pth_outputs):
        np_pth_output = pth_output.detach().numpy()
        max_absdiff = np.abs(ort_output - np_pth_output)
        allclose = np.allclose(ort_output, np_pth_output, atol=1.e-5)
        results.append((max_absdiff, allclose))

    for result in results:
        if result[1] == False:
            return False

    return True
