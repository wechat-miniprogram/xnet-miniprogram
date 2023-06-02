from typing import Union, Callable, Tuple, Dict, Optional, Type

import torch.nn as nn

import nncs.nn.intrinsic as nni
import nncs.fuser.fusion
from nncs.quant.utils import get_combined_dict
from .fusion import fuse_deconv_bn_eval, fuse_linear_bn_eval


def fuse_conv_bn(conv, bn):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert (
        conv.training == bn.training
    ), "Conv and BN both must be in the same mode (train or eval)."

    fused_module_class_map = {
        nn.Conv1d: nni.ConvBn1d,
        nn.Conv2d: nni.ConvBn2d,
        nn.Conv3d: nni.ConvBn3d,
    }

    if conv.training:
        assert (
            bn.num_features == conv.out_channels
        ), "Output channel of Conv2d must match num_features of BatchNorm2d"
        assert bn.affine, "Only support fusing BatchNorm2d with affine set to True"
        assert (
            bn.track_running_stats
        ), "Only support fusing BatchNorm2d with tracking_running_stats set to True"
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((conv, bn))
            )
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_linear_bn(linear, bn):
    assert (
        linear.training == bn.training
    ), "Conv and BN both must be in the same mode (train or eval)."

    fused_module_class_map = {
        nn.Linear: nni.LinearBn1d,
    }

    if linear.training:
        assert (
            bn.num_features == linear.out_features
        ), "Output channel of Conv2d must match num_features of BatchNorm2d"
        assert bn.affine, "Only support fusing BatchNorm2d with affine set to True"
        assert (
            bn.track_running_stats
        ), "Only support fusing BatchNorm2d with tracking_running_stats set to True"
        fused_module_class = fused_module_class_map.get((type(linear)), None)
        if fused_module_class is not None:
            return fused_module_class(linear, bn)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((linear, bn))
            )
    else:
        return fuse_linear_bn_eval(linear, bn)


def fuse_deconv_bn(deconv, bn):
    fused_module_class_map = {nn.ConvTranspose2d: nni.ConvTransposeBn2d}

    if deconv.training:
        fused_module_class = fused_module_class_map.get((type(deconv)), None)
        if fused_module_class is not None:
            return fused_module_class(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_bn_relu(deconv, bn, relu):
    if deconv.training:
        map_to_fused_module_train = {nn.ConvTranspose2d: nni.ConvTransposeBnReLU2d}
        fused_module = map_to_fused_module_train.get(type(deconv), None)
        if fused_module is not None:
            return fused_module(deconv, bn, relu)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((deconv, bn, relu))
            )
    else:
        map_to_fused_module_eval = {nn.ConvTranspose2d: nni.ConvTransposeReLU2d}
        fused_module = map_to_fused_module_eval.get(type(deconv), None)
        if fused_module is not None:
            fused_deconv = fuse_deconv_bn_eval(deconv, bn)
            return fused_module(fused_deconv, relu)
        else:
            raise NotImplementedError(
                "Cannot fuse eval modules: {}".format((deconv, bn, relu))
            )


def fuse_deconv_bn_relu6(deconv, bn, relu6):
    if deconv.training:
        map_to_fused_module_train = {nn.ConvTranspose2d: nni.ConvTransposeBnReLU62d}
        fused_module = map_to_fused_module_train.get(type(deconv), None)
        if fused_module is not None:
            return fused_module(deconv, bn, relu6)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((deconv, bn, relu6))
            )
    else:
        map_to_fused_module_eval = {nn.ConvTranspose2d: nni.ConvTransposeReLU62d}
        fused_module = map_to_fused_module_eval.get(type(deconv), None)
        if fused_module is not None:
            fused_deconv = fuse_deconv_bn_eval(deconv, bn)
            return fused_module(fused_deconv, relu6)
        else:
            raise NotImplementedError(
                "Cannot fuse eval modules: {}".format((deconv, bn, relu6))
            )


def fuse_conv_bn_relu(conv, bn, relu):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert (
        conv.training == bn.training == relu.training
    ), "Conv and BN both must be in the same mode (train or eval)."
    fused_module: Optional[Type[nn.Sequential]] = None
    if conv.training:
        map_to_fused_module_train = {
            nn.Conv1d: nni.ConvBnReLU1d,
            nn.Conv2d: nni.ConvBnReLU2d,
            nn.Conv3d: nni.ConvBnReLU3d,
        }
        assert (
            bn.num_features == conv.out_channels
        ), "Output channel of Conv must match num_features of BatchNorm"
        assert bn.affine, "Only support fusing BatchNorm with affine set to True"
        assert (
            bn.track_running_stats
        ), "Only support fusing BatchNorm with tracking_running_stats set to True"
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((conv, bn, relu))
            )
    else:
        map_to_fused_module_eval = {
            nn.Conv1d: nni.ConvReLU1d,
            nn.Conv2d: nni.ConvReLU2d,
            nn.Conv3d: nni.ConvReLU3d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = nncs.fuser.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError(
                "Cannot fuse eval modules: {}".format((conv, bn, relu))
            )


def fuse_conv_bn_relu6(conv, bn, relu6):
    assert (
        conv.training == bn.training == relu6.training
    ), "Conv and BN both must be in the same mode (train or eval)."
    fused_module: Optional[Type[nn.Sequential]] = None
    if conv.training:
        map_to_fused_module_train = {
            nn.Conv2d: nni.ConvBnReLU62d,
        }
        assert (
            bn.num_features == conv.out_channels
        ), "Output channel of Conv must match num_features of BatchNorm"
        assert bn.affine, "Only support fusing BatchNorm with affine set to True"
        assert (
            bn.track_running_stats
        ), "Only support fusing BatchNorm with tracking_running_stats set to True"
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, relu6)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((conv, bn, relu6))
            )
    else:
        map_to_fused_module_eval = {
            nn.Conv2d: nni.ConvReLU62d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = nncs.fuser.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu6)
        else:
            raise NotImplementedError(
                "Cannot fuse eval modules: {}".format((conv, bn, relu6))
            )


def fuse_conv_bn_clip8(conv, bn, clip8):
    assert (
        conv.training == bn.training == clip8.training
    ), "Conv and BN both must be in the same mode (train or eval)."
    fused_module: Optional[Type[nn.Sequential]] = None
    if conv.training:
        if isinstance(clip8, nni.custom_op.learnable_relu._LearnableClip):
            map_to_fused_module_train = {
                nn.Conv2d: nni.ConvBnLearnableClip82d,
            }
        else:
            assert False

        assert (
            bn.num_features == conv.out_channels
        ), "Output channel of Conv must match num_features of BatchNorm"
        assert bn.affine, "Only support fusing BatchNorm with affine set to True"
        assert (
            bn.track_running_stats
        ), "Only support fusing BatchNorm with tracking_running_stats set to True"
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, clip8)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((conv, bn, clip8))
            )
    else:
        if isinstance(clip8, nni.custom_op.learnable_relu._LearnableClip):
            map_to_fused_module_eval = {
                nn.Conv2d: nni.ConvLearnableClip82d,
            }
        else:
            assert False

        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = nncs.fuser.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, clip8)
        else:
            raise NotImplementedError(
                "Cannot fuse eval modules: {}".format((conv, bn, clip8))
            )


DEFAULT_OP_LIST_TO_FUSER_METHOD: Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): nni.ConvReLU1d,
    (nn.Conv2d, nn.ReLU): nni.ConvReLU2d,
    (nn.Conv3d, nn.ReLU): nni.ConvReLU3d,
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nn.ReLU): nni.LinearReLU,
    (nn.BatchNorm2d, nn.ReLU): nni.BNReLU2d,
    (nn.BatchNorm3d, nn.ReLU): nni.BNReLU3d,
}

CUSTOM_OP_LIST_TO_FUSER_METHOD: Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU6): fuse_conv_bn_relu6,
    (nn.Conv2d, nn.ReLU6): nni.ConvReLU62d,
    (nn.Linear, nn.ReLU6): nni.LinearReLU6,
    (
        nn.Conv2d,
        nn.BatchNorm2d,
        nni.custom_op.learnable_relu._LearnableClip,
    ): fuse_conv_bn_clip8,
    (nn.ConvTranspose2d, nn.BatchNorm2d): fuse_deconv_bn,
    (nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU): fuse_deconv_bn_relu,
    (nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU6): fuse_deconv_bn_relu6,
    (nn.ConvTranspose2d, nn.ReLU): nni.ConvTransposeReLU2d,
    (nn.ConvTranspose2d, nn.ReLU6): nni.ConvTransposeReLU62d,
}

# (nn.quantized.FloatFunctional, nni.custom_op.learnable_relu._LearnableClip): nni.FloatFunctionalClip8,


def get_fuser_method(op_list, additional_fuser_method_mapping=None):
    """Get fuser method for the given list of module types,
    return None if fuser method does not exist
    """
    if additional_fuser_method_mapping is None:
        additional_fuser_method_mapping = {}
    all_mappings = get_combined_dict(
        DEFAULT_OP_LIST_TO_FUSER_METHOD, additional_fuser_method_mapping
    )
    all_mappings = get_combined_dict(all_mappings, CUSTOM_OP_LIST_TO_FUSER_METHOD)
    fuser_method = all_mappings.get(op_list, None)
    assert fuser_method is not None, "did not find fuser method for: {} ".format(
        op_list
    )

    return fuser_method
