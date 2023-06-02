import torch.nn as nn


class MaxPool2d(nn.MaxPool2d):
    _FLOAT_MODULE = nn.MaxPool2d

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        qconfig=None,
    ):
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.input_fake_quant = qconfig.activation()

    def forward(self, input):
        quant_input = self.input_fake_quant(input)
        return super(MaxPool2d, self).forward(quant_input)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by nncs.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qconfig = mod.qconfig
        q_mod = mod

        qat_mod = cls(
            q_mod.kernel_size,
            q_mod.stride,
            q_mod.padding,
            q_mod.dilation,
            q_mod.return_indices,
            q_mod.ceil_mode,
            qconfig=qconfig,
        )
        return qat_mod


class AvgPool2d(nn.AvgPool2d):
    _FLOAT_MODULE = nn.AvgPool2d

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        qconfig=None,
    ):
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.input_fake_quant = qconfig.activation()

    def forward(self, input):
        quant_input = self.input_fake_quant(input)
        return super(AvgPool2d, self).forward(quant_input)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by nncs.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qconfig = mod.qconfig
        q_mod = mod

        qat_mod = cls(
            q_mod.kernel_size,
            q_mod.stride,
            q_mod.padding,
            q_mod.ceil_mode,
            q_mod.count_include_pad,
            q_mod.divisor_override,
            qconfig=qconfig,
        )
        return qat_mod


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    _FLOAT_MODULE = nn.AdaptiveAvgPool2d

    def __init__(self, output_size, qconfig=None):
        super().__init__(output_size)
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.input_fake_quant = qconfig.activation()

    def forward(self, input):
        quant_input = self.input_fake_quant(input)
        return super(AdaptiveAvgPool2d, self).forward(quant_input)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by nncs.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qconfig = mod.qconfig
        q_mod = mod

        qat_mod = cls(q_mod.output_size, qconfig=qconfig)
        return qat_mod
