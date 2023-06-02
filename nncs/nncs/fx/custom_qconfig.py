import torch

from nncs.fake_quantize.fake_quantize_torch1100 import FakeQuantize
from nncs.quant.qconfig import QConfig
from nncs.fake_quantize.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)


def tflite_qconfig(nbits=8, averaging_constant=0.01, affine=True):
    assert nbits <= 8 and nbits >= 2
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-(2 ** (nbits - 1)),
            quant_max=2 ** (nbits - 1) - 1,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine if affine else torch.per_tensor_symmetric,
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