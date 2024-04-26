import warnings

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op
from torch.onnx.symbolic_opset11 import clamp


@parse_args("v", "v", "v", "i", "i", "i")
def fake_quantize_per_channel_affine(g, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127):
    if quant_min not in [0, -128, -127] or quant_max not in [127, 255]:
        raise RuntimeError(
            "ONNX defines [0, 255] for quint8 and [-128/-127, 127] for qint8, got [{}, {}]".format(quant_min, quant_max)
        )

    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx["Byte"])
    else:
        zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx["Char"])

    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis),
        scale, zero_point, axis_i=axis)


def relu6(g, input):
    relu = input
    dtype = input.type().scalarType()
    if dtype is None:
        dtype = 6  # float
    else:
        dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    min_val = g.op("Constant", value_t=torch.tensor(0, dtype=sym_help.scalar_type_to_pytorch_type[dtype]))
    max_val = g.op("Constant", value_t=torch.tensor(6, dtype=sym_help.scalar_type_to_pytorch_type[dtype]))
    return clamp(g, relu, min_val, max_val)


def register_extra_symbolics(opset=11):
    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This function will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    register_op('fake_quantize_per_channel_affine', fake_quantize_per_channel_affine, '', opset)
    register_op('relu6', relu6, '', opset)
