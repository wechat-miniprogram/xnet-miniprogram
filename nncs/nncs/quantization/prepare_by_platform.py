from typing import Dict, List, Tuple, Callable, Any
from enum import Enum
from functools import partial
from packaging import version

import torch

from nncs.nncs_network import NNCSNetwork
from nncs.quant.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS
from nncs.dynamic_graph.graph_builder import ModelInputInfo
from nncs.quant.qconfig import QConfig

from nncs.fake_quantize.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    MovingAveragePercentileObserver,
    MovingAveragePerChannelPercentileObserver,
)

if version.parse(torch.__version__) >= version.parse("1.10.0"):
    # pylint: disable=wildcard-import
    from nncs.fake_quantize.fake_quantize_torch1100 import *
else:
    # pylint: disable=wildcard-import
    from nncs.fake_quantize.fake_quantize import *

from nncs.fake_quantize.lpt import LptFakeQuantize


class PlatformType(Enum):
    XNet = "XNet"
    XNetInt7 = "XNetInt7"
    Tensorrt = "Tensorrt"
    Lpt = "Lpt"


def tflite_qconfig(nbits=8, averaging_constant=0.01):
    assert nbits <= 8 and nbits >= 2
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-(2 ** (nbits - 1)),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            averaging_constant=averaging_constant,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-(2 ** (nbits - 1) - 1),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            averaging_constant=averaging_constant,
        ),
    )
    return qconfig


def tensorrt_qconfig(nbits=8, averaging_constant=0.01):
    assert nbits <= 8 and nbits >= 2
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-(2 ** (nbits - 1)),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
            averaging_constant=averaging_constant,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-(2 ** (nbits - 1)),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            averaging_constant=averaging_constant,
        ),
    )
    return qconfig


def lpt_qconfig(nbits=8, averaging_constant=0.01):
    assert nbits <= 8 and nbits >= 2
    qconfig = QConfig(
        activation=LptFakeQuantize.with_args(
            forward_observer=MovingAverageMinMaxObserver,
            backward_observer=MovingAveragePercentileObserver,
            quant_min=-(2 ** (nbits - 1)),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            averaging_constant=averaging_constant,
            backward_params={"percentile": 0.98},
        ),
        weight=LptFakeQuantize.with_args(
            forward_observer=MovingAveragePerChannelMinMaxObserver,
            backward_observer=MovingAveragePerChannelPercentileObserver,
            quant_min=-(2 ** (nbits - 1)),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            averaging_constant=averaging_constant,
            backward_params={"percentile": 0.99},
        ),
    )
    return qconfig


PlatformTable8bit = {
    PlatformType.XNetInt7: partial(tflite_qconfig, nbits=7),
    PlatformType.XNet: partial(tflite_qconfig, nbits=8),
    PlatformType.Tensorrt: partial(tensorrt_qconfig, nbits=8),
    PlatformType.Lpt: partial(lpt_qconfig, nbits=8),
}


def prepare_by_platform(
    model: torch.nn.Module,
    input_shapes: List[Tuple],
    deploy_platform: PlatformType,
    qat_module_mappings: Dict[Callable, Any] = None,
):

    if qat_module_mappings is None:
        qat_module_mappings = DEFAULT_QAT_MODULE_MAPPINGS

    input_info_list = []
    for input_shape in input_shapes:
        input_info = ModelInputInfo(shape=input_shape)
        input_info_list.append(input_info)

    network = NNCSNetwork(model, input_info_list)
    specification = PlatformTable8bit[deploy_platform]()

    if deploy_platform in [PlatformType.XNet, PlatformType.XNetInt7, PlatformType.Lpt]:
        merged_graph, merged_module, merged_nodenames = network.merge_modules()

        qat_modules = network.convert_to_qat_modules(
            merged_graph,
            merged_module,
            merged_nodenames,
            qat_module_mappings,
            specification,
        )
        if deploy_platform in [PlatformType.XNet, PlatformType.XNetInt7]:
            cs = network.apply_constriant(merged_graph)
        else:
            cs = None

        _ = network.insert_activation_fq(
            merged_graph, qat_modules, merged_nodenames, cs
        )

    elif deploy_platform in [PlatformType.Tensorrt]:
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
        network.cvt_to_trt_module(qat_module_mappings, qconfig=specification)
    else:
        assert False, "Unsupported Platform"

    return network
