import torch
from .observer import (
    MovingAverageMinMaxObserver,
    HistogramObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from .fake_quantize_base import _is_per_channel, _is_per_tensor, FakeQuantizeBase


class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enable` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype


    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        **observer_kwargs
    ):
        super().__init__()
        assert (
            quant_min <= quant_max
        ), "quant_min must be less than or equal to quant_max"
        self.quant_min = quant_min
        self.quant_max = quant_max
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        self.activation_post_process = observer(**observer_kwargs)
        assert (
            torch.iinfo(self.activation_post_process.dtype).min <= quant_min
        ), "quant_min out of bound"
        assert (
            quant_max <= torch.iinfo(self.activation_post_process.dtype).max
        ), "quant_max out of bound"
        self.register_buffer("scale", torch.tensor([1.0]))
        self.register_buffer("zero_point", torch.tensor([0]))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        assert _is_per_channel(self.qscheme) or _is_per_tensor(self.qscheme), (
            "Only per channel and per tensor quantization are supported in fake quantize"
            + " got qscheme: "
            + str(self.qscheme)
        )
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1 and self.observer_opened[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                Xq = torch.fake_quantize_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point.to(torch.int),
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                )
            else:
                Xq = torch.fake_quantize_per_tensor_affine(
                    X,
                    float(self.scale),
                    int(self.zero_point),
                    self.quant_min,
                    self.quant_max,
                )
            return Xq
        else:
            return X

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, "
            "quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, "
            "scale={}, zero_point={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.quant_min,
                self.quant_max,
                self.dtype,
                self.qscheme,
                self.ch_axis,
                self.scale,
                self.zero_point,
            )
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "zero_point"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if "zero_point" in key:
                    val = val.long()
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


default_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
default_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False,
)
default_per_channel_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    reduce_range=False,
    ch_axis=0,
)
default_histogram_fake_quant = FakeQuantize.with_args(
    observer=HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
