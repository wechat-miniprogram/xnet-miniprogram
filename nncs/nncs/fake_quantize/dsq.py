from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .observer import (
    MovingAverageMinMaxObserver,
    _with_args,
)
from .fake_quantize_base import _is_per_channel, _is_per_tensor


class DSQSignWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(
            x > 0, torch.Tensor([1]).to(x.device), torch.Tensor([-1]).to(x.device)
        )

    @staticmethod
    def backward(ctx, g):
        return g


class BinActive(torch.autograd.Function):
    """
    Binarize the input activations and calculate the mean across channel dimension.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # size = input.size()
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class TanhGradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.tanh(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = 1.0 - torch.pow(torch.tanh(input), 2)
        grad_input = torch.clamp(grad_input, 0, 2)
        return grad_input


class DSQ_func(torch.autograd.Function):
    def __init__(self):
        super(DSQ_func, self).__init__()

    @staticmethod
    def forward(
        ctx,
        X,
        _alpha,
        scale,
        zero_point,
        _range_min,
        _range_max,
        quant_min,
        quant_max,
        is_per_channel,
        ch_axis,
    ):
        range_margin = _range_max - _range_min
        range_min = -(range_margin / 2.0)
        range_max = range_margin / 2.0
        if is_per_channel:
            weight_shape = [1] * len(X.shape)
            weight_shape[0] = -1
            range_min = range_min.reshape(weight_shape)
            range_max = range_max.reshape(weight_shape)
            scale = scale.reshape(weight_shape)
            zero_point = zero_point.reshape(weight_shape)
            alpha = _alpha.reshape(weight_shape)
        else:
            alpha = _alpha

        ap = torch.clamp(alpha, 1e-7, 0.5)
        masked_ap_l = alpha.lt(1e-7)
        masked_ap_u = alpha.gt(0.5)

        idx = torch.round((X - range_min) / scale)
        s = 1.0 / (1 - ap)
        pian_s_pian_ap = s**2
        pian_s_pian_a = pian_s_pian_ap
        pian_s_pian_a[masked_ap_l] = 0.0
        pian_s_pian_a[masked_ap_u] = 0.0

        invScale = 1.0 / scale

        k = torch.log(2.0 / ap - 1.0) * invScale
        pian_k_pian_ap = -2.0 / (ap * (2.0 - ap)) * invScale
        pian_k_pian_a = pian_k_pian_ap
        pian_s_pian_a[masked_ap_l] = 0.0
        pian_s_pian_a[masked_ap_u] = 0.0

        m = range_min + (idx) * scale
        diff = X - m
        stype = torch.tanh(k * diff)
        pian_stype_pian_k = (1 - stype**2) * (X - m)
        phi = s * stype

        sgn = DSQSignWithGradient.apply(phi)
        pian_phi_pian_a = pian_s_pian_a * stype + s * pian_stype_pian_k * pian_k_pian_a
        int8_idx = (idx + (sgn + 1) * 0.5) + quant_min

        up_bound = quant_max - zero_point
        lw_bound = quant_min - zero_point
        masked_l = int8_idx.lt(lw_bound)
        masked_u = int8_idx.gt(up_bound)
        value = 0.5 * s * (1 - stype**2) * k
        value[masked_l] = 0.0
        value[masked_u] = 0.0
        value = scale * value

        pian_int8_idx_pian_a = 0.5 * pian_phi_pian_a
        pian_int8_idx_pian_a[masked_l] = 0.0
        pian_int8_idx_pian_a[masked_u] = 0.0
        pian_y_pian_a = scale * pian_int8_idx_pian_a

        ctx.save_for_backward(value, pian_y_pian_a, _alpha)

        if is_per_channel:
            X = torch.fake_quantize_per_channel_affine(
                X, scale.flatten(), zero_point.flatten(), ch_axis, quant_min, quant_max
            ).detach()
        else:
            X = torch.fake_quantize_per_tensor_affine(
                X, float(scale), int(zero_point), quant_min, quant_max
            ).detach()
        return X

    @staticmethod
    def backward(ctx, grad_output):
        value, pian_y_pian_a, alpha = ctx.saved_tensors

        grad_input = grad_output.clone() * value
        grad_alpha = grad_output.clone() * pian_y_pian_a
        grad_alpha = grad_alpha.reshape(alpha.shape[0], -1)
        grad_alpha = torch.sum(grad_alpha, dim=1)

        return grad_input, grad_alpha, None, None, None, None, None, None, None, None


class DSQBase(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("debug", torch.tensor([1], dtype=torch.uint8))

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

    with_args = classmethod(_with_args)


class DSQ_fakeQuantize(DSQBase):
    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        channel_len=-1,
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
        self.register_buffer("debug", torch.tensor([0]))
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

        if channel_len != -1:
            assert (
                isinstance(channel_len, int) and channel_len > 0
            ), "Channel size must be a positive integer."
            self.alpha = Parameter(
                torch.tensor([0.2] * channel_len, requires_grad=True)
            )
        else:
            self.alpha = Parameter(torch.tensor([0.2], requires_grad=True))

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
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
                range_min = self.activation_post_process.min_vals
                range_max = self.activation_post_process.max_vals
                X = DSQ_func.apply(
                    X,
                    self.alpha,
                    self.scale,
                    self.zero_point,
                    range_min,
                    range_max,
                    self.quant_min,
                    self.quant_max,
                    self.is_per_channel,
                    self.ch_axis,
                )
                # X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point,
                #                                           self.ch_axis, self.quant_min, self.quant_max)
            else:
                range_min = self.activation_post_process.min_val
                range_max = self.activation_post_process.max_val
                X = DSQ_func.apply(
                    X,
                    self.alpha,
                    self.scale,
                    self.zero_point,
                    range_min,
                    range_max,
                    self.quant_min,
                    self.quant_max,
                    self.is_per_channel,
                    self.ch_axis,
                )
                # X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                #                                          int(self.zero_point), self.quant_min,
                #                                          self.quant_max)
        return X

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, "
            "quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, "
            "scale={}, zero_point={}, alpha={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.quant_min,
                self.quant_max,
                self.dtype,
                self.qscheme,
                self.ch_axis,
                self.scale,
                self.zero_point,
                self.alpha.data,
            )
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(DSQ_fakeQuantize, self)._save_to_state_dict(
            destination, prefix, keep_vars
        )
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point
        destination[prefix + "alpha"] = self.alpha

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
        local_state = ["scale", "zero_point", "alpha"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if "zero_point" in key:
                    val = val.long()
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(DSQ_fakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def disable_fake_quant(mod):
    if isinstance(mod, DSQBase):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    if isinstance(mod, DSQBase):
        mod.enable_fake_quant()


def disable_observer(mod):
    if isinstance(mod, DSQBase):
        mod.disable_observer()


def enable_observer(mod):
    if isinstance(mod, DSQBase):
        mod.enable_observer()
