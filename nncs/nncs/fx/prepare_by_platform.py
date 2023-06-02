from enum import Enum
from functools import partial
from typing import Dict, Callable, Union, Any

import torch
import torch.nn as nn
from torch.fx._symbolic_trace import Tracer, GraphModule

from nncs.quant.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS
from nncs.fx.custom_quantizer import (
    XNetModelQuantizer,
    TRTModelQuantizer,
)
from nncs.fx.constraint_solver import ConstraintSolver
from nncs.checker.onnx2torch_verifier import verify_o2t_convert
from onnx2torch import convert
from .onnx_quantizer import ONNXQuantizer
from .custom_qconfig import tflite_qconfig, tensorrt_qconfig


class PlatformType(Enum):
    XNet = "XNet"
    XNetInt7 = "XNetInt7"
    Tensorrt = "Tensorrt"
    XNetSymInt8 = "XNetSymInt8"


PlatformTable8bit = {
    PlatformType.XNetSymInt8: partial(tflite_qconfig, nbits=8, affine=False),
    PlatformType.XNetInt7: partial(tflite_qconfig, nbits=7),
    PlatformType.XNet: partial(tflite_qconfig, nbits=8),
    PlatformType.Tensorrt: partial(tensorrt_qconfig, nbits=8)
}


class NNCSTracer(Tracer):
    def is_leaf_module(self, m: nn.Module, mod_qual_name: str) -> bool:
        module_name = m.__module__
        if module_name.startswith("torch.nn"):
            return not isinstance(m, (torch.nn.Sequential, ))
        elif module_name.startswith("nncs.nn.intrinsic"):
            return not isinstance(m, torch.nn.Sequential)
        elif module_name.startswith("nncs.nn.modules"):
            if module_name.startswith("nncs.nn.modules.transformer"):
                return False
            else:
                return not isinstance(m, torch.nn.Sequential)
        elif module_name.startswith("torchvision.ops.stochastic_depth"):
            return not isinstance(m, torch.nn.Sequential)
        elif module_name.startswith("nncs.fx.no_quant"):
            return True
        else:
            pass
        return False


def prepare_by_platform(
    module_or_path: Union[str, torch.nn.Module],
    deploy_platform: PlatformType,
    qat_module_mappings: Dict[Callable, Any] = None,
    **kwargs):
    if qat_module_mappings is None:
        qat_module_mappings = DEFAULT_QAT_MODULE_MAPPINGS

    if isinstance(module_or_path, str):
        onnx_path = module_or_path
        graph_model = convert(onnx_path)

        if kwargs.get('verify_o2t', False):
            passed = verify_o2t_convert(onnx_path, graph_model)
            if not passed:
                assert(False), "onnx2torch verify failed"

        prepared = prepare_onnx(graph_model, deploy_platform,
                                qat_module_mappings, **kwargs)
    elif isinstance(module_or_path, torch.nn.Module):
        pytorch_nn_module = module_or_path
        prepared = prepare_nnmodule(pytorch_nn_module, deploy_platform,
                                    qat_module_mappings, **kwargs)
    else:
        assert(False), "Wrong input type: {}".format(type(module_or_path))
    return prepared


def prepare_onnx(
    graph_module,
    deploy_platform: PlatformType,
    qat_module_mappings: Dict[Callable, Any] = None,
    **kwargs
):
    specification = PlatformTable8bit[deploy_platform]()

    quantizer = ONNXQuantizer()

    if deploy_platform in [PlatformType.XNet, PlatformType.XNetInt7, PlatformType.XNetSymInt8]:
        constraint_solver = ConstraintSolver()
    else:
        constraint_solver = None

    prepared = quantizer.prepare(graph_module,
            constraint_solver,
            specification,
            qat_module_mappings,
            **kwargs)

    return prepared


def prepare_nnmodule(
    model: torch.nn.Module,
    deploy_platform: PlatformType,
    qat_module_mappings: Dict[Callable, Any] = None,
    **kwargs
):
    tracer = NNCSTracer()
    if 'concrete_args' in kwargs:
        graph = tracer.trace(model, concrete_args=kwargs['concrete_args'])
    else:
        graph = tracer.trace(model, concrete_args=None)
    name = (
        model.__class__.__name__
        if isinstance(model, torch.nn.Module)
        else model.__name__
    )
    graph_module = GraphModule(tracer.root, graph, name)

    specification = PlatformTable8bit[deploy_platform]()

    if deploy_platform in [PlatformType.XNet, PlatformType.XNetInt7, PlatformType.XNetSymInt8]:
        quantizer = XNetModelQuantizer()

        if deploy_platform in [PlatformType.XNet, PlatformType.XNetInt7, PlatformType.XNetSymInt8]:
            constraint_solver = ConstraintSolver()
        else:
            constraint_solver = None

        prepared = quantizer.prepare(
            graph_module,
            constraint_solver,
            specification,
            qat_module_mappings,
            **kwargs
        )
    elif deploy_platform in [PlatformType.Tensorrt]:
        quantizer = TRTModelQuantizer()
        constraint_solver = None

        import torch.nn as nn
        from nncs.nn.trt.qat.conv import Conv2d
        from nncs.nn.trt.qat.linear import Linear
        from nncs.nn.trt.qat.pooling import MaxPool2d, AvgPool2d, AdaptiveAvgPool2d

        qat_module_mappings = {
            nn.Conv2d: Conv2d,
            nn.Linear: Linear,
            nn.MaxPool2d: MaxPool2d,
            nn.AvgPool2d: AvgPool2d,
            nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
        }

        prepared = quantizer.prepare(
            graph_module,
            constraint_solver,
            specification,
            qat_module_mappings,
            **kwargs
        )
    else:
        assert False, "Unsupported PlatformType"

    return prepared
