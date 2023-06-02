from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from .observer import (
    MovingAverageMinMaxObserver,
    _with_args,
)
from .fake_quantize import FakeQuantize


def _is_per_channel(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]


def _is_per_tensor(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


class LptBase(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))

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


class LptFunc(torch.autograd.Function):
    def __init__(self):
        super(LptFunc, self).__init__()

    @staticmethod
    def forward(ctx, X, f_s, f_zp, quant_min, quant_max, is_per_channel, ch_axis):
        if is_per_channel:
            Xq = torch.fake_quantize_per_channel_affine(
                X, f_s, f_zp, ch_axis, quant_min, quant_max
            )
        else:
            Xq = torch.fake_quantize_per_tensor_affine(
                X, float(f_s), int(f_zp), quant_min, quant_max
            )
        return Xq

    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


class LptFakeQuantize(LptBase):
    def __init__(
        self,
        forward_observer=MovingAverageMinMaxObserver,
        backward_observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        **observer_kwargs
    ):
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max

        # backward need to be unsymmetry
        backward_fq_ob_kwargs = {}
        for key in observer_kwargs:
            if key == "qscheme":
                if observer_kwargs[key] == torch.per_channel_symmetric:
                    backward_fq_ob_kwargs[key] = torch.per_channel_affine
                elif observer_kwargs[key] == torch.per_tensor_symmetric:
                    backward_fq_ob_kwargs[key] = torch.per_tensor_affine
                else:
                    backward_fq_ob_kwargs[key] = observer_kwargs[key]
            else:
                backward_fq_ob_kwargs[key] = observer_kwargs[key]
        backward_fq_ob_kwargs["include_zero"] = False
        if "backward_params" in observer_kwargs:
            backward_fq_ob_kwargs.update(observer_kwargs["backward_params"])
            del observer_kwargs["backward_params"]
            del backward_fq_ob_kwargs["backward_params"]

        self.backward_fq = FakeQuantize(
            observer=backward_observer,
            quant_min=quant_min,
            quant_max=quant_max,
            **backward_fq_ob_kwargs
        )

        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max

        self.forward_ob = forward_observer(**observer_kwargs)
        self.ch_axis = (
            self.forward_ob.ch_axis if hasattr(self.forward_ob, "ch_axis") else -1
        )
        self.qscheme = self.forward_ob.qscheme

        self.register_buffer("scale", torch.tensor([1.0]))
        self.register_buffer("zero_point", torch.tensor([0]))

        self.is_per_channel = _is_per_channel(self.qscheme)

        def backward_hook(module, grad_input, grad_output):
            with torch.no_grad():
                _grad_input_detach = grad_input[0].detach()
                module.vis_handler.plot(_grad_input_detach, module.name)
                grad_input = self.backward_fq(_grad_input_detach)
                grad_input_detach = grad_input.detach()
                module.vis_handler.plot_dc(
                    grad_input_detach, _grad_input_detach, module.name
                )
                dc = 1.0 - F.cosine_similarity(
                    grad_input_detach.flatten(), _grad_input_detach.flatten(), 0, 1e-8
                )
                module.dc = dc.detach().cpu().numpy()
            return (grad_input,)

        from packaging import version

        if version.parse(torch.__version__) >= version.parse("1.10"):
            self.register_full_backward_hook(backward_hook)
        else:
            self.register_backward_hook(backward_hook)

    def calculate_qparams(self):
        return self.forward_ob.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.forward_ob(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)

        # if self.fake_quant_enabled[0] == 1:
        #     Xq = LptFunc.apply(X, self.scale, self.zero_point,
        #         self.quant_min, self.quant_max, self.is_per_channel, self.ch_axis)
        #     return Xq
        # else:
        #     return X

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                Xq = torch.fake_quantize_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
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
            "quant_min={}, quant_max={}, ch_axis={}, "
            "scale={}, zero_point={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.quant_min,
                self.quant_max,
                self.ch_axis,
                self.scale,
                self.zero_point,
            )
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(LptFakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
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
        super(LptFakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def disable_fake_quant(mod):
    if isinstance(mod, LptBase):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    if isinstance(mod, LptBase):
        mod.enable_fake_quant()


def disable_observer(mod):
    if isinstance(mod, LptBase):
        mod.disable_observer()


def enable_observer(mod):
    if isinstance(mod, LptBase):
        mod.enable_observer()
