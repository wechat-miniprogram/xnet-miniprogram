from .conv_fused import (
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBn1d,
    ConvBn2d,
    ConvBnLearnableReLU62d,
    ConvBnReLU62d,
    ConvBnLearnableClip82d,
)

# from .deconv_fused import ConvTransposeBn2d, ConvTransposeBnReLU2d, ConvTransposeBnReLU62d
from .deconv_fused import ConvTransposeBn2d


def update_bn_stats(mod):
    if type(mod) in set(
        [
            ConvBnReLU1d,
            ConvBnReLU2d,
            ConvBn1d,
            ConvBn2d,
            ConvBnLearnableReLU62d,
            ConvBnReLU62d,
            ConvBnLearnableClip82d,
            ConvTransposeBn2d,
        ]
    ):
        mod.update_bn_stats()


def freeze_bn_stats(mod):
    if type(mod) in set(
        [
            ConvBnReLU1d,
            ConvBnReLU2d,
            ConvBn1d,
            ConvBn2d,
            ConvBnLearnableReLU62d,
            ConvBnReLU62d,
            ConvBnLearnableClip82d,
            ConvTransposeBn2d,
        ]
    ):
        mod.freeze_bn_stats()
