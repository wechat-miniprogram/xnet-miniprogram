import unittest
import torch
import torch.nn as nn

import nncs
from nncs.fx.prepare_by_platform import prepare_by_platform, PlatformType
from model_zoo.fcn import VGGNet, FCN32s


class TestQuantizeDeConvModel(unittest.TestCase):
    def test_model0(self):
        class Model(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                inner_channels = 256
                self.deconv = nn.ConvTranspose2d(
                    inner_channels // 4, inner_channels // 4, 2, 2
                )
                self.bn = nn.BatchNorm2d(inner_channels // 4)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.relu(self.bn(self.deconv(x)))

        model = Model()
        model_prepared = prepare_by_platform(model, PlatformType.XNet)
        dummy_input = torch.randn(2, 64, 32, 32, device="cpu")
        model_prepared.train()
        model_prepared.apply(nncs.fake_quantize.enable_observer)
        output = model_prepared(dummy_input)
        loss = output.sum()

        loss.backward()
        model_prepared.apply(nncs.fake_quantize.disable_observer)
        model_prepared.eval()
        torch.onnx.export(
            model_prepared,
            dummy_input,
            "model_prepared.onnx",
            export_params=True,
            verbose=False,
            training=False,
            opset_version=13,
            do_constant_folding=True,
        )

    def test_model1(self):
        vgg_model = VGGNet(requires_grad=True, pretrained=False)
        model = FCN32s(pretrained_net=vgg_model, n_class=2)
        model_prepared = prepare_by_platform(model, PlatformType.XNet)
        dummy_input = torch.randn(2, 3, 256, 256, device="cpu")
        model_prepared.train()
        model_prepared.apply(nncs.fake_quantize.enable_observer)
        output = model_prepared(dummy_input)
        loss = output.sum()

        loss.backward()
        model_prepared.apply(nncs.fake_quantize.disable_observer)
        model_prepared.eval()
        torch.onnx.export(
            model_prepared,
            dummy_input,
            "model_prepared.onnx",
            export_params=True,
            verbose=False,
            training=False,
            opset_version=13,
            do_constant_folding=True,
        )
