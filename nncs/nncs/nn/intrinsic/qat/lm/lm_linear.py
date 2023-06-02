import torch.nn.functional as F

import nncs
from nncs.nn import qat as nnqat
from nncs.nn import intrinsic as nni
from nncs.nn.intrinsic.custom_op import LearnableReLU6


class LinearReLU6(nnqat.Linear, nni._FusedModule):
    _FLOAT_MODULE = nni.LinearReLU6

    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super(LinearReLU6, self).__init__(in_features, out_features, bias, qconfig)

        if isinstance(
            self.qconfig.weight(),
            nncs.fake_quantize.c_learnable_fake_quantize._LearnableFakeQuantize,
        ) or isinstance(self.qconfig.weight(), nncs.fake_quantize.dsq.DSQ_fakeQuantize):
            self.weight_fake_quant = self.qconfig.weight(channel_len=out_features)
        else:
            self.weight_fake_quant = self.qconfig.weight()

        self.relu = LearnableReLU6(inplace=False)

    def forward(self, input):
        return self.relu(
            F.linear(input, self.weight_fake_quant(self.weight), self.bias)
        )

    @classmethod
    def from_float(cls, mod):
        return super(LinearReLU6, cls).from_float(mod)
