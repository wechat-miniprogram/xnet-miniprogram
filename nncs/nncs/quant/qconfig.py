from collections import namedtuple
from typing import Union

import torch
import torch.nn as nn
import nncs

from distutils.version import LooseVersion

if LooseVersion(torch.__version__) < LooseVersion("1.10.0"):
    from nncs.fake_quantize.fake_quantize import *
else:
    from nncs.fake_quantize.fake_quantize_torch1100 import *

from nncs.fake_quantize.observer import *
from nncs.fake_quantize.c_learnable_fake_quantize import _LearnableFakeQuantize
from nncs.fake_quantize.dsq import DSQ_fakeQuantize


class QConfig(namedtuple("QConfig", ["activation", "weight"])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfig(activation=MinMaxObserver.with_args(dtype=torch.qint8),
      weight=default_observer.with_args(dtype=torch.qint8))
    """

    def __new__(cls, activation, weight):
        # catch common mistakes
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "QConfig received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super(QConfig, cls).__new__(cls, activation, weight)


default_qconfig = QConfig(activation=default_observer, weight=default_weight_observer)

default_debug_qconfig = QConfig(
    weight=default_weight_observer, activation=default_debug_observer
)

default_per_channel_qconfig = QConfig(
    activation=default_observer, weight=default_per_channel_weight_observer
)


class QConfigDynamic(namedtuple("QConfigDynamic", ["activation", "weight"])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """

    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        # catch common mistakes
        if isinstance(weight, nn.Module):
            raise ValueError(
                "QConfigDynamic received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super(QConfigDynamic, cls).__new__(cls, activation, weight)


default_dynamic_qconfig = QConfigDynamic(
    activation=default_dynamic_quant_observer, weight=default_weight_observer
)
float16_dynamic_qconfig = QConfigDynamic(
    activation=PlaceholderObserver.with_args(dtype=torch.float16),
    weight=PlaceholderObserver.with_args(dtype=torch.float16),
)
per_channel_dynamic_qconfig = QConfigDynamic(
    activation=default_dynamic_quant_observer,
    weight=default_per_channel_weight_observer,
)

# TODO: this is weight only quant, change this to QConfigWeightOnly
# or remove the QConfigDynamic later
float_qparams_weight_only_qconfig = QConfigDynamic(
    activation=default_placeholder_observer, weight=default_float_qparams_observer
)

default_qat_qconfig = QConfig(
    activation=default_fake_quant, weight=default_weight_fake_quant
)

default_weight_only_qconfig = QConfig(
    activation=torch.nn.Identity, weight=default_weight_fake_quant
)
default_activation_only_qconfig = QConfig(
    activation=default_fake_quant, weight=torch.nn.Identity
)


def get_default_qconfig(backend="fbgemm"):
    if backend == "fbgemm":
        qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=True),
            weight=default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=False),
            weight=default_weight_observer,
        )
    else:
        qconfig = default_qconfig
    return qconfig


def get_default_qat_qconfig(backend="fbgemm"):
    # Histogram observer is too slow for quantization aware training
    if backend == "fbgemm":
        qconfig = QConfig(
            activation=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                reduce_range=True,
            ),
            weight=default_per_channel_weight_fake_quant,
        )
    elif backend == "qnnpack":
        qconfig = QConfig(
            activation=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                reduce_range=False,
            ),
            weight=default_weight_fake_quant,
        )
    else:
        qconfig = default_qat_qconfig
    return qconfig


def get_tflite_qat_qconfig_nbits(nbits=8, averaging_constant=0.01):
    assert nbits <= 8 and nbits >= 2
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-(2 ** (nbits - 1)),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=0,
            averaging_constant=averaging_constant,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-(2 ** (nbits - 1) - 1),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=-1,
            averaging_constant=averaging_constant,
        ),
    )
    return qconfig


def get_default_qat_lfq_qconfig(averaging_constant=0.01):
    qconfig = QConfig(
        activation=_LearnableFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            reduce_range=0,
            qscheme=torch.per_tensor_affine,
            dtype=torch.qint8,
            averaging_constant=averaging_constant,
        ),
        weight=_LearnableFakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=-1,
            ch_axis=0,
            averaging_constant=averaging_constant,
        ),
    )
    return qconfig


def get_qat_dsq_qconfig(averaging_constant=0.01):
    qconfig = QConfig(
        activation=DSQ_fakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            reduce_range=0,
            qscheme=torch.per_tensor_affine,
            dtype=torch.qint8,
            averaging_constant=averaging_constant,
        ),
        weight=DSQ_fakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=-1,
            ch_axis=0,
            averaging_constant=averaging_constant,
        ),
    )
    return qconfig


def assert_valid_qconfig(
    qconfig: Union[QConfig, QConfigDynamic], mod: torch.nn.Module
) -> None:
    is_conv_transpose_mod = (
        isinstance(mod, torch.nn.ConvTranspose1d)
        or isinstance(mod, torch.nn.ConvTranspose2d)
        or isinstance(mod, torch.nn.ConvTranspose3d)
    )
    if is_conv_transpose_mod:
        example_observer = qconfig.weight()
        is_per_channel = isinstance(
            example_observer, nncs.quantization.PerChannelMinMaxObserver
        ) or isinstance(
            example_observer, nncs.quantization.MovingAveragePerChannelMinMaxObserver
        )
        assert (
            not is_per_channel
        ), "Per channel weight observer is not supported yet for ConvTranspose{n}d."
