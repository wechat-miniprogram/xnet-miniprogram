import copy
import torch


def fuse_deconv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, groups):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    n_conv_weight = []
    step = conv_w.shape[1] // groups
    bn_step = bn_rm.shape[0] // groups
    for i in range(groups):
        p_conv_w = conv_w[:, i * step : (i + 1) * step].clone()
        p_bn_w = bn_w[i * bn_step : (i + 1) * bn_step].clone()
        p_bn_var_rsqrt = bn_var_rsqrt[i * bn_step : (i + 1) * bn_step].clone()

        p_conv_w = p_conv_w * (p_bn_w * p_bn_var_rsqrt).reshape(
            [-1] + [1] * (len(conv_w.shape) - 1)
        )
        n_conv_weight.append(p_conv_w)

    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    conv_w = torch.cat(n_conv_weight, dim=1).permute(1, 0, 2, 3)

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def fuse_deconv_bn_eval(deconv, bn):
    assert not (deconv.training or bn.training), "Fusion only for eval!"
    fused_deconv = copy.deepcopy(deconv)

    with torch.no_grad():
        permuted_weight = fused_deconv.weight.permute(1, 0, 2, 3)
        merge_pw, merge_bias = fuse_deconv_bn_weights(
            permuted_weight,
            fused_deconv.bias,
            bn.running_mean,
            bn.running_var,
            bn.eps,
            bn.weight,
            bn.bias,
            deconv.groups,
        )
        fused_deconv.weight = merge_pw
        fused_deconv.bias = merge_bias

    return fused_deconv


def fuse_conv_bn_eval(conv, bn):
    assert not (conv.training or bn.training), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(conv_w.shape) - 1)
    )
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def fuse_linear_bn_eval(linear, bn):
    assert not (linear.training or bn.training), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(
        fused_linear.weight,
        fused_linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return fused_linear


def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    linear_w = linear_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(linear_w.shape) - 1)
    )
    linear_b = (linear_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(linear_w), torch.nn.Parameter(linear_b)
