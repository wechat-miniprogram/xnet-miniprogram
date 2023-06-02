import nncs
from nncs.nn import qat as nnqat
from nncs.nn import intrinsic as nni
from nncs.nn.intrinsic.custom_op import LearnableReLU6


class ConvReLU62d(nnqat.Conv2d, nni._FusedModule):
    _FLOAT_MODULE = nni.ConvReLU62d

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
        super(ConvReLU62d, self).__init__(
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
            self.weight_fake_quant = self.qconfig.weight(channel_len=out_channels)
        else:
            self.weight_fake_quant = self.qconfig.weight()

        self.relu = LearnableReLU6(inplace=False)

    def forward(self, input):
        return self.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        )
