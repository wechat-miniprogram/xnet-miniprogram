import re
import torch
from .fake_quantize_base import FakeQuantizeBase
from .dsq import DSQ_fakeQuantize
from .lpt import LptBase


def _is_fake_quant_script_module(mod):
    """Returns true if given mod is an instance of FakeQuantize script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return name == "nncs.fake_quantize.fake_quantize.FakeQuantize"
    return False


def disable_fake_quant(mod):
    if isinstance(
        mod, (FakeQuantizeBase, DSQ_fakeQuantize, LptBase)
    ) or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    if isinstance(
        mod, (FakeQuantizeBase, DSQ_fakeQuantize, LptBase)
    ) or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()


def disable_observer(mod):
    if isinstance(
        mod, (FakeQuantizeBase, DSQ_fakeQuantize, LptBase)
    ) or _is_fake_quant_script_module(mod):
        mod.disable_observer()


def enable_observer(mod):
    if isinstance(
        mod, (FakeQuantizeBase, DSQ_fakeQuantize, LptBase)
    ) or _is_fake_quant_script_module(mod):
        mod.enable_observer()
