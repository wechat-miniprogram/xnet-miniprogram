from collections import OrderedDict
from inspect import Signature, Parameter
from typing import List

import torch

from nncs.dynamic_graph.patch_pytorch import register_operator
from nncs.dynamic_graph.graph_builder import ModelInputInfo, create_mock_tensor
from nncs.utils import is_tensor, objwalk
from nncs.common.utils.logger import logger as nncs_logger


@register_operator()
def nncs_model_input(tensor: "torch.Tensor"):
    return tensor.clone()


def wrap_nncs_model_inputs_with_objwalk(model_args, model_kwargs):
    model_args = objwalk(model_args, is_tensor, nncs_model_input)
    model_kwargs = objwalk(model_kwargs, is_tensor, nncs_model_input)
    return model_args, model_kwargs


class InputInfoWrapManager:
    INPUTS_MISMATCH_WARNING_TEXT = (
        "Compression with regards to this input may occur incorrectly. Make sure "
        "you call the compressed model with inputs that correspond to what NNCS was "
        "configured to expect (either via NNCS config's input_infos, or custom"
        "dummy_forward_fn/wrap_inputs_fn parameters), or that you know what you are "
        "doing. This warning will not be shown again."
    )
    ARGS_INPUTS_MISMATCH_FORMAT_STRING = (
        "Inputs mismatch - could not find arg with idx {} in NNCS-wrapped model "
        "input args! " + INPUTS_MISMATCH_WARNING_TEXT
    )
    KWARGS_INPUTS_MISMATCH_FORMAT_STRING = (
        "Inputs mismatch - could not find kwarg '{}' in NNCS-wrapped model input "
        "kwargs! " + INPUTS_MISMATCH_WARNING_TEXT
    )

    def __init__(
        self,
        input_infos: List[ModelInputInfo],
        fwd_signature: Signature,
        module_ref_for_device: torch.nn.Module = None,
    ):
        self._module_ref_for_device = module_ref_for_device
        arg_iis_list = [ii for ii in input_infos if ii.keyword is None]
        kwarg_iis_list = [
            (ii.keyword, ii) for ii in input_infos if ii.keyword is not None
        ]
        kwarg_iis = OrderedDict()
        arg_iis = tuple(arg_iis_list)
        for kw, ii in kwarg_iis_list:
            kwarg_iis[kw] = ii
        bound_params = fwd_signature.bind(*arg_iis, **kwarg_iis)

        self._fwd_params_to_input_infos_odict = bound_params.arguments
        self._fwd_signature = fwd_signature  # type: Signature

    def set_device(self, device: str):
        self._device = device

    def wrap_inputs(self, model_args, model_kwargs):
        bound_model_params = self._fwd_signature.bind(*model_args, **model_kwargs)
        for param_name in self._fwd_params_to_input_infos_odict:
            param_kind = self._fwd_signature.parameters[param_name].kind
            if (
                param_kind is Parameter.VAR_POSITIONAL
                or param_kind is Parameter.VAR_KEYWORD
            ):
                nncs_logger.warning(
                    "An input_info tensor was bound to a *args or **kwargs variadic parameter in the"
                    "forward's signature! This is currently unsupported by NNCS. Input compression may "
                    "be incorrect."
                )
                # Currently won't support input info mapping to *args or **kwargs-mapped parameters
                continue

            if param_name not in bound_model_params.arguments:
                nncs_logger.warning(
                    "A call to a compressed model's forward occured without one of the params"
                    "specified in input_infos! Input compression may be incorrect. Trying to recover "
                    "by wrapping the default value for the parameter."
                )
                bound_model_params.apply_defaults()

            potential_tensor = bound_model_params.arguments[param_name]
            if potential_tensor is not None:
                bound_model_params.arguments[param_name] = nncs_model_input(
                    bound_model_params.arguments[param_name]
                )
            else:
                # Default was None - cannot wrap as-is. Will wrap a dummy tensor as specified in
                # input infos - will conserve the call order of nncs_model_input nodes,
                # and the post-hooks for the input node will execute. The result won't go anywhere, though.
                nncs_logger.warning(
                    "Wrapping a dummy tensor for input {}".format(param_name)
                )
                info_for_missing_input = self._fwd_params_to_input_infos_odict[
                    param_name
                ]
                device = "cuda"
                if self._module_ref_for_device is not None:
                    device = next(self._module_ref_for_device.parameters()).device
                dummy_tensor = create_mock_tensor(info_for_missing_input, device)
                _ = nncs_model_input(dummy_tensor)

        return bound_model_params.args, bound_model_params.kwargs


MODEL_INPUT_OP_NAME = nncs_model_input._original_op.__name__
