from abc import ABC, abstractmethod
import torch
from torch.nn import Module
from .observer import (
    _with_args,
)


def _is_per_channel(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]


def _is_per_tensor(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


class FakeQuantizeBase(ABC, Module):
    r"""Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(self):
        super().__init__()
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_opened", torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_observer(self):
        self.enable_observer(False)

    @torch.jit.export
    def open_observer(self, enabled: bool = True) -> None:
        self.observer_opened[0] = 1 if enabled else 0

    @torch.jit.export
    def close_observer(self):
        self.open_observer(False)

    with_args = classmethod(_with_args)


class FixedQParamsFakeQuantize(FakeQuantizeBase):
    """Simulate quantize and dequantize with fixed quantization
    parameters in training time. Only per tensor quantization
    is supported.
    Args:
        `scale` (float): fixed scale for the fake quantize module
        `zero_point` (int): fixed zero point for the fake quantize module
        `dtype`, `qscheme`, `quant_min`, `quant_max`
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        scale,
        zero_point,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
    ):
        super().__init__()
        assert (
            quant_min <= quant_max
        ), "quant_min should be less than or equal to quant_max"
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("zero_point", torch.tensor([zero_point]))
        self.dtype = dtype
        self.qscheme = qscheme
        assert _is_per_tensor(self.qscheme), (
            "Only per tensor quantization is supported"
            + " FixedQParamsFakeQuantize module, got qscheme:"
            + str(self.qscheme)
        )

    def forward(self, X):
        if self.fake_quant_enabled[0] == 1:
            X = torch.fake_quantize_per_tensor_affine(
                X,
                float(self.scale),
                int(self.zero_point),
                self.quant_min,
                self.quant_max,
            )
        return X

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, scale={}, zero_point={}, "
            "dtype={}, quant_min={}, quant_max={}, qscheme={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.scale,
                self.zero_point,
                self.dtype,
                self.quant_min,
                self.quant_max,
                self.qscheme,
            )
        )


# TODO(future PR): remove these defaults and enforce activation functions
# to explicitly specify their output range
default_symmetric_fixed_qparams_fake_quant = FixedQParamsFakeQuantize.with_args(
    scale=2.0 / 256.0, zero_point=128, dtype=torch.quint8, quant_min=0, quant_max=255
)
default_affine_fixed_qparams_fake_quant = FixedQParamsFakeQuantize.with_args(
    scale=1.0 / 256.0, zero_point=0, dtype=torch.quint8, quant_min=0, quant_max=255
)
