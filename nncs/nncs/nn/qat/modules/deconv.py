import torch.nn as nn
import torch.nn.functional as F
from nncs.nn.intrinsic import ConvTransposeReLU2d, ConvTransposeReLU62d


class ConvTranspose2d(nn.ConvTranspose2d):
    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        assert qconfig, "qconfig must be provided for QAT module"
        self.weight_fake_quant = qconfig.weight(ch_axis=0)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        # permuted_weight -> out_c//group, in_c, kh, kw
        permuted_weight = self.weight.permute(1, 0, 2, 3)
        permuted_weight = permuted_weight.reshape(
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        weight_fq = self.weight_fake_quant(permuted_weight)
        weight_fq = weight_fq.reshape(
            self.out_channels // self.groups,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        weight_fq = weight_fq.permute(1, 0, 2, 3)

        o = F.conv_transpose2d(
            input,
            weight_fq,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

        return o

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig

        if type(mod) == ConvTransposeReLU2d or type(mod) == ConvTransposeReLU62d:
            mod = mod[0]

        qat_deconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_deconv.weight = mod.weight
        qat_deconv.bias = mod.bias
        return qat_deconv
