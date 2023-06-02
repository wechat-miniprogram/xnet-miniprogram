from collections import OrderedDict
from typing import Dict


from torch.quantization.fx.quantization_types import Pattern
from torch.quantization.fx.pattern_utils import QuantizeHandler

# pattern for conv bn fusion
CUSTOM_FUSION_PATTERNS = OrderedDict()


def register_fusion_pattern(pattern):
    def insert(fn):
        CUSTOM_FUSION_PATTERNS[pattern] = fn
        return fn

    return insert


def get_custom_fusion_patterns() -> Dict[Pattern, QuantizeHandler]:
    return CUSTOM_FUSION_PATTERNS
