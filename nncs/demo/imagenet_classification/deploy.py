import torch
import os, sys

abs_path = os.path.dirname(os.path.abspath(__file__))
nncs_home = os.path.join(abs_path, "../../")
sys.path.insert(0, nncs_home)

import nncs
from nncs.fx.prepare_by_platform import prepare_by_platform, PlatformType

input_shape = (1, 3, 224, 224)
model = prepare_by_platform("mobilenet-v2-71dot82.onnx", PlatformType.XNet)
ckpt = torch.load("mobilenet_v2_best_lr.tar")
out = model.load_state_dict(ckpt["state_dict"], strict=False)
print(out)

model.cpu()
model.eval()
model.apply(nncs.fake_quantize.disable_observer)

mock_t = torch.rand(input_shape).float()
torch.onnx.export(model, mock_t, "mobilenetv2_qat.onnx", opset_version=13, do_constant_folding=True, export_params=True)
