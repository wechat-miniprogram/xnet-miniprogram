import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

import nncs
from nncs.nn import intrinsic as nni
from nncs.nn import qat as nnqat


_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


class _ConvTransposeBnNd(nn.modules.conv._ConvTransposeNd, nni._FusedModule):

    _version = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        eps=1e-05,
        momentum=0.1,
        freeze_bn=False,
        qconfig=None,
        dim=2,
    ):
        nn.modules.conv._ConvTransposeNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            False,
            padding_mode,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        if isinstance(
            self.qconfig.weight(),
            nncs.fake_quantize.c_learnable_fake_quantize._LearnableFakeQuantize,
        ) or isinstance(self.qconfig.weight(), nncs.fake_quantize.dsq.DSQ_fakeQuantize):
            self.weight_fake_quant = self.qconfig.weight(
                channel_len=out_channels, ch_axis=0
            )
        else:
            self.weight_fake_quant = self.qconfig.weight(ch_axis=0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_bn_parameters()

        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ConvTransposeBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _forward(self, input, output_size=None):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std

        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1

        permuted_weight = self.weight.permute(1, 0, 2, 3)
        permuted_weight = permuted_weight.reshape(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size
        )

        scaled_weight = self.weight_fake_quant(
            permuted_weight * scale_factor.reshape(weight_shape)
        )
        scaled_weight = scaled_weight.reshape(
            self.out_channels // self.groups, self.in_channels, *self.kernel_size
        )
        scaled_weight = scaled_weight.permute(1, 0, 2, 3)

        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)

        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        deconv = F.conv_transpose2d(
            input,
            scaled_weight,
            zero_bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        scale_factor = torch.where(
            scale_factor == 0,
            torch.tensor(self.bn.eps).to(scale_factor.device),
            scale_factor,
        )

        channel_shape = [1] * len(deconv.shape)
        channel_shape[1] = -1
        deconv_orig = deconv / scale_factor.reshape(channel_shape)

        if self.bias is not None:
            deconv_orig = deconv_orig + self.bias.reshape(channel_shape)

        deconv_o = self.bn(deconv_orig)

        return deconv_o

    def extra_repr(self):
        return super(_ConvTransposeBnNd, self).extra_repr()

    def forward(self, input, output_size=None):
        return self._forward(input, output_size)

    def train(self, mode=True):
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

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
        super(_ConvTransposeBnNd, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_float(cls, mod):
        qconfig = mod.qconfig
        deconv, bn = mod[0], mod[1]
        qat_deconvbn = cls(
            deconv.in_channels,
            deconv.out_channels,
            deconv.kernel_size,
            deconv.stride,
            deconv.padding,
            deconv.dilation,
            deconv.groups,
            deconv.bias is not None,
            deconv.padding_mode,
            bn.eps,
            bn.momentum,
            False,
            qconfig,
        )
        qat_deconvbn.weight = deconv.weight
        qat_deconvbn.bias = deconv.bias
        qat_deconvbn.bn.weight = bn.weight
        qat_deconvbn.bn.bias = bn.bias
        qat_deconvbn.bn.running_mean = bn.running_mean
        qat_deconvbn.bn.running_var = bn.running_var
        qat_deconvbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_deconvbn


class ConvTransposeBn2d(_ConvTransposeBnNd, nn.ConvTranspose2d):
    _FLOAT_MODULE = nni.ConvTransposeBn2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        eps=1e-05,
        momentum=0.1,
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        _ConvTransposeBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=2,
        )


class ConvTransposeBnReLU2d(ConvTransposeBn2d):
    _FLOAT_MODULE = nni.ConvTransposeBnReLU2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        eps=1e-05,
        momentum=0.1,
        freeze_bn=False,
        qconfig=None,
    ):
        super(ConvTransposeBnReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
        )

    def forward(self, input, output_size=None):
        return F.relu(ConvTransposeBnReLU2d._forward(self, input, output_size))


class ConvTransposeBnReLU62d(ConvTransposeBn2d):
    _FLOAT_MODULE = nni.ConvTransposeBnReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        eps=1e-05,
        momentum=0.1,
        freeze_bn=False,
        qconfig=None,
    ):
        super(ConvTransposeBnReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
        )

    def forward(self, input, output_size=None):
        return F.relu6(ConvTransposeBnReLU62d._forward(self, input, output_size))


class ConvTransposeReLU2d(nnqat.ConvTranspose2d, nni._FusedModule):
    _FLOAT_MODULE = nni.ConvTransposeReLU2d

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
        super(ConvTransposeReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig

        if isinstance(
            self.qconfig.weight(),
            nncs.fake_quantize.c_learnable_fake_quantize._LearnableFakeQuantize,
        ) or isinstance(self.qconfig.weight(), nncs.fake_quantize.dsq.DSQ_fakeQuantize):
            self.weight_fake_quant = self.qconfig.weight(
                channel_len=out_channels, ch_axis=0
            )
        else:
            self.weight_fake_quant = self.qconfig.weight(ch_axis=0)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )

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

        o = F.relu(
            F.conv_transpose2d(
                input,
                weight_fq,
                self.bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        )
        return o

    @classmethod
    def from_float(cls, mod):
        return super(ConvTransposeReLU2d, cls).from_float(mod)


class ConvTransposeReLU62d(nnqat.ConvTranspose2d, nni._FusedModule):
    _FLOAT_MODULE = nni.ConvTransposeReLU62d

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
        super(ConvTransposeReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig

        if isinstance(
            self.qconfig.weight(),
            nncs.fake_quantize.c_learnable_fake_quantize._LearnableFakeQuantize,
        ) or isinstance(self.qconfig.weight(), nncs.fake_quantize.dsq.DSQ_fakeQuantize):
            self.weight_fake_quant = self.qconfig.weight(
                channel_len=out_channels, ch_axis=0
            )
        else:
            self.weight_fake_quant = self.qconfig.weight(ch_axis=0)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
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
        o = F.relu6(
            F.conv_transpose2d(
                input,
                weight_fq,
                self.bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        )
        return o

    @classmethod
    def from_float(cls, mod):
        return super(ConvTransposeReLU62d, cls).from_float(mod)
