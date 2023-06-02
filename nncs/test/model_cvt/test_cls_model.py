import unittest
import torch

import nncs
from nncs.common.utils.logger import logger as nncs_logger
from nncs.fx.prepare_by_platform import prepare_by_platform, PlatformType

from version import GITHUB_RES


class TestQuantizeClsModel(unittest.TestCase):
    def test_model_torchvision(self):

        entrypoints = torch.hub.list(GITHUB_RES, force_reload=False)
        for entrypoint in entrypoints:

            nncs_logger.info("testing %s" % entrypoint)
            if "deeplab" in entrypoint or "fcn" in entrypoint:
                if entrypoint in ["googlenet", "inception_v3"]:
                    model_to_quantize = torch.hub.load(
                        GITHUB_RES,
                        entrypoint,
                        pretrained=False,
                        pretrained_backbone=False,
                        aux_logits=False,
                    )
                else:
                    model_to_quantize = torch.hub.load(
                        GITHUB_RES,
                        entrypoint,
                        pretrained=False,
                        pretrained_backbone=False,
                    )
            else:
                if entrypoint in ["googlenet", "inception_v3"]:
                    model_to_quantize = torch.hub.load(
                        GITHUB_RES, entrypoint, pretrained=False, aux_logits=False
                    )
                else:
                    model_to_quantize = torch.hub.load(
                        GITHUB_RES, entrypoint, pretrained=False
                    )

            if entrypoint in ["inception_v3"]:
                dummy_input = torch.randn(2, 3, 299, 299, device="cpu")
            else:
                dummy_input = torch.randn(2, 3, 224, 224, device="cpu")
            model_prepared = prepare_by_platform(model_to_quantize, PlatformType.XNet)
            model_prepared.train()
            model_prepared.apply(nncs.fake_quantize.enable_observer)
            output = model_prepared(dummy_input)
            if isinstance(output, torch.Tensor):
                loss = output.sum()
            elif isinstance(output, (dict)):
                loss = output["out"].sum()
            else:
                assert False

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
