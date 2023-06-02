import torch
import torch.nn as nn
from torch.nn import (
    Conv1d,
    Conv2d,
    Conv3d,
    ReLU,
    Linear,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    ReLU6,
)
from ..custom_op.learnable_relu import _LearnableClip


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class ConvReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        assert (
            type(conv) == Conv1d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(type(conv), type(relu))
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        assert (
            type(conv) == Conv2d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(type(conv), type(relu))
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        assert (
            type(conv) == Conv3d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(type(conv), type(relu))
        super().__init__(conv, relu)


class ConvReLU62d(_FusedModule):
    def __init__(self, conv, relu6):
        assert (
            type(conv) == Conv2d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(type(conv), type(relu6))
        super().__init__(conv, relu6)


class ConvLearnableClip82d(_FusedModule):
    def __init__(self, conv, clip8):
        assert (
            type(conv) == Conv2d and type(clip8) == _LearnableClip
        ), "Incorrect types for input modules{}{}".format(type(conv), type(clip8))
        super().__init__(conv, clip8)


class LinearReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, relu):
        assert (
            type(linear) == Linear and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(type(linear), type(relu))
        super().__init__(linear, relu)


class LinearReLU6(_FusedModule):
    def __init__(self, linear, relu6):
        assert (
            type(linear) == Linear and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(type(linear), type(relu6))
        super().__init__(linear, relu6)


class LinearBn1d(_FusedModule):
    def __init__(self, linear, bn):
        assert (
            type(linear) == Linear and type(bn) == BatchNorm1d
        ), "Incorrect types for input modules{}{}".format(type(linear), type(bn))
        super().__init__(linear, bn)


class ConvBn1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        assert (
            type(conv) == Conv1d and type(bn) == BatchNorm1d
        ), "Incorrect types for input modules{}{}".format(type(conv), type(bn))
        super().__init__(conv, bn)


class ConvBn2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        assert (
            type(conv) == Conv2d and type(bn) == BatchNorm2d
        ), "Incorrect types for input modules{}{}".format(type(conv), type(bn))
        super(ConvBn2d, self).__init__(conv, bn)


class ConvBnReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        assert (
            type(conv) == Conv1d and type(bn) == BatchNorm1d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(relu)
        )
        super().__init__(conv, bn, relu)


class ConvBnReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        assert (
            type(conv) == Conv2d and type(bn) == BatchNorm2d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(relu)
        )
        super().__init__(conv, bn, relu)


class ConvTransposeBn2d(_FusedModule):
    def __init__(self, deconv, bn):
        assert (
            type(deconv) == nn.ConvTranspose2d and type(bn) == BatchNorm2d
        ), "Incorrect types for input modules{}{}".format(type(deconv), type(bn))
        super().__init__(deconv, bn)


class ConvTransposeBnReLU2d(_FusedModule):
    def __init__(self, deconv, bn, relu):
        assert (
            type(deconv) == nn.ConvTranspose2d
            and type(bn) == BatchNorm2d
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(deconv), type(bn), type(relu)
        )
        super().__init__(deconv, bn, relu)


class ConvTransposeBnReLU62d(_FusedModule):
    def __init__(self, deconv, bn, relu6):
        assert (
            type(deconv) == nn.ConvTranspose2d
            and type(bn) == BatchNorm2d
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(deconv), type(bn), type(relu6)
        )
        super().__init__(deconv, bn, relu6)


class ConvTransposeReLU2d(_FusedModule):
    def __init__(self, deconv, relu):
        assert (
            type(deconv) == nn.ConvTranspose2d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(type(deconv), type(relu))
        super().__init__(deconv, relu)


class ConvTransposeReLU62d(_FusedModule):
    def __init__(self, deconv, relu6):
        assert (
            type(deconv) == nn.ConvTranspose2d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(type(deconv), type(relu6))
        super().__init__(deconv, relu6)


class ConvBn3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        assert (
            type(conv) == Conv3d and type(bn) == BatchNorm3d
        ), "Incorrect types for input modules{}{}".format(type(conv), type(bn))
        super().__init__(conv, bn)


class ConvBnReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        assert (
            type(conv) == Conv3d and type(bn) == BatchNorm3d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(relu)
        )
        super().__init__(conv, bn, relu)


class ConvBnReLU62d(_FusedModule):
    def __init__(self, conv, bn, relu6):
        assert (
            type(conv) == Conv2d and type(bn) == BatchNorm2d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(relu6)
        )
        super().__init__(conv, bn, relu6)


class ConvBnLearnableClip82d(_FusedModule):
    def __init__(self, conv, bn, clip8):
        assert (
            type(conv) == Conv2d
            and type(bn) == BatchNorm2d
            and type(clip8) == _LearnableClip
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(clip8)
        )
        super().__init__(conv, bn, clip8)


class BNReLU2d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        assert (
            type(batch_norm) == BatchNorm2d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(type(batch_norm), type(relu))
        super().__init__(batch_norm, relu)


class BNReLU3d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        assert (
            type(batch_norm) == BatchNorm3d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(type(batch_norm), type(relu))
        super().__init__(batch_norm, relu)


# class FloatFunctionalClip8(_FusedModule):
#    def __init__(self, floatFunctional, clip8):
#        assert type(floatFunctional) == nn.quantized.FloatFunctional and type(clip8) == _LearnableClip, \
#            'Incorrect types for input modules{}{}'.format(
#                type(floatFunctional), type(clip8))
#        super().__init__(floatFunctional, clip8)
