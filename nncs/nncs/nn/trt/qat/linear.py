import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super().__init__(in_features, out_features, bias)
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.input_fake_quant = qconfig.activation()
        self.weight_fake_quant = qconfig.weight()

    def forward(self, input):
        quant_input = self.input_fake_quant(input)
        quant_weight = self.weight_fake_quant(self.weight)
        return F.linear(quant_input, quant_weight, self.bias)

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

        qat_linear = cls(
            q_mod.in_features,
            q_mod.out_features,
            bias=q_mod.bias is not None,
            qconfig=qconfig,
        )
        qat_linear.weight = q_mod.weight
        qat_linear.bias = q_mod.bias
        return qat_linear
