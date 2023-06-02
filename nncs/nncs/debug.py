import logging
import warnings
from typing import List, Dict

from torch.nn import Module
from nncs.common.utils.logger import logger as nncs_logger


DEBUG_LOG_DIR = "./nncs_debug"


def is_debug():
    return nncs_logger.getEffectiveLevel() == logging.DEBUG


def set_debug_log_dir(dir_: str):
    global DEBUG_LOG_DIR
    DEBUG_LOG_DIR = dir_


class CallCountTracker:
    def __init__(self, name):
        self.name = name
        self.call_counts = {}

    def init_with_key_list(self, key_list: List):
        self.call_counts = {key: 0 for key in key_list}
        nncs_logger.debug("{} tracker: registered {} entries".format(self.name, len(self.call_counts)))

    def register_call(self, key, counts=None):
        if key not in self.call_counts:
            warnings.warn("DEBUG: {} tracker: called an unregistered module: {}".format(self.name, key))
            return
        if counts is None:
            self.call_counts[key] += 1
        else:
            self.call_counts[key] = counts

    def get_never_called_keys(self) -> List[str]:
        return [k for k, v in self.call_counts.items() if v == 0]

    def get_overcalled_keys_with_call_counts(self) -> Dict[str, int]:
        return {k: v for k, v in self.call_counts.items() if v > 1}

    def get_total_call_count(self) -> int:
        if self.call_counts:
            return sum(self.call_counts.values())
        return 0

    def reset(self):
        for key in self.call_counts:
            self.call_counts[key] = 0


class DebugInterface:
    def pre_forward_actions(self, module: Module):
        raise NotImplementedError

    def post_forward_actions(self, module: Module):
        raise NotImplementedError

    def init_actual(self, owner_model):
        raise NotImplementedError


def debuggable_forward(forward_func):
    def decorated(self, *args, **kwargs):
        if self.debug_interface is not None:
            self.debug_interface.pre_forward_actions(module=self)
        retval = forward_func(self, *args, **kwargs)
        if self.debug_interface is not None:
            self.debug_interface.post_forward_actions(module=self)
        return retval

    return decorated


class CombinedDebugInterface(DebugInterface):
    def __init__(self):
        self._interfaces = []  # type: List[DebugInterface]

    def add_interface(self, interface: 'DebugInterface'):
        self._interfaces.append(interface)

    def init_actual(self, owner_model: 'NNCSNetwork'):
        for interface in self._interfaces:
            interface.init_actual(owner_model)

    def pre_forward_actions(self, module: Module):
        for interface in self._interfaces:
            interface.pre_forward_actions(module)

    def post_forward_actions(self, module: Module):
        for interface in self._interfaces:
            interface.post_forward_actions(module)
