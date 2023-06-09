from collections import OrderedDict
from typing import Callable, Any, List, Optional
from copy import deepcopy

import torch


class ModelInputInfo:
    FILLER_TYPE_ONES = "ones"
    FILLER_TYPE_ZEROS = "zeros"
    FILLER_TYPE_RANDOM = "random"
    FILLER_TYPE_SPEC = "spec"
    FILLER_TYPES = [
        FILLER_TYPE_ONES,
        FILLER_TYPE_ZEROS,
        FILLER_TYPE_RANDOM,
        FILLER_TYPE_SPEC,
    ]

    def __init__(
        self,
        shape: List[int],
        type_str: str = "float",
        keyword=None,
        filler=None,
        filler_tensor=None,
    ):
        self.shape = shape
        self.type = self._string_to_torch_type(type_str)
        self.keyword = keyword
        if filler is None:
            self.filler = self.FILLER_TYPE_ONES
        else:
            self.filler = filler
            if self.filler not in self.FILLER_TYPES:
                raise RuntimeError("Unknown input filler type: {}".format(filler))
        self.filler_tensor = filler_tensor

    @staticmethod
    def _string_to_torch_type(string):
        if string == "long":
            return torch.long
        return torch.float32

    @staticmethod
    def torch_type_to_string(dtype: torch.dtype):
        if dtype is torch.long:
            return "long"
        return "float"

    def is_integer_input(self):
        return self.type != torch.float32

    def __eq__(self, other):
        return self.type == other.type and self.keyword == other.keyword


def create_input_infos(config) -> List[ModelInputInfo]:
    input_infos = config.get("input_info", [])
    if isinstance(input_infos, dict):
        return [
            ModelInputInfo(
                input_infos.get("sample_size"),
                input_infos.get("type"),
                input_infos.get("keyword"),
                input_infos.get("filler"),
            ),
        ]
    if isinstance(input_infos, list):
        if not input_infos:
            return [ModelInputInfo([1, 3, 224, 224])]
        return [
            ModelInputInfo(
                info_dict.get("sample_size"),
                info_dict.get("type"),
                info_dict.get("keyword"),
                info_dict.get("filler"),
            )
            for info_dict in input_infos
        ]
    raise RuntimeError(
        "Invalid input_infos specified in config - should be either dict or list of dicts"
    )


def create_mock_tensor(input_info: ModelInputInfo, device: str):
    args = {"size": input_info.shape, "dtype": input_info.type, "device": device}
    if input_info.filler == ModelInputInfo.FILLER_TYPE_ZEROS:
        return torch.zeros(**args)
    if input_info.filler == ModelInputInfo.FILLER_TYPE_ONES:
        return torch.ones(**args)
    if input_info.filler == ModelInputInfo.FILLER_TYPE_RANDOM:
        return torch.rand(**args)
    if input_info.filler == ModelInputInfo.FILLER_TYPE_SPEC:
        return input_info.filler_tensor.to(device)
    raise RuntimeError


class GraphBuilder:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def build_graph(
        self,
        model: torch.nn.Module,
        context_to_use: Optional["TracingContext"] = None,
        as_eval: bool = False,
    ) -> "NNCSGraph":

        sd = deepcopy(model.state_dict())
        from nncs.dynamic_graph.context import TracingContext

        if context_to_use is None:
            context_to_use = TracingContext()

        from nncs.utils import training_mode_switcher

        context_to_use.base_module_thread_local_replica = model
        with context_to_use as _ctx:
            with torch.no_grad():
                if as_eval:
                    with training_mode_switcher(model, is_training=False):
                        self.custom_forward_fn(model)
                else:
                    self.custom_forward_fn(model)
        model.load_state_dict(sd)

        if isinstance(model, PostGraphBuildActing):
            model.post_build_graph_actions()
        return context_to_use.graph


class PostGraphBuildActing:
    def post_build_graph_actions(self):
        pass


def create_dummy_forward_fn(
    input_infos: List[ModelInputInfo], with_input_tracing=False, wrap_inputs_fn=None
):
    from nncs.dynamic_graph.input_wrapping import wrap_nncs_model_inputs_with_objwalk

    def default_dummy_forward_fn(model):
        try:
            device = next(model.parameters()).device
        except Exception:  # pylint: disable=broad-except
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        args_list = [
            create_mock_tensor(info, device)
            for info in input_infos
            if info.keyword is None
        ]
        kwargs = OrderedDict()
        for info in input_infos:
            if info.keyword is not None:
                kwargs[info.keyword] = create_mock_tensor(info, device)
        args = tuple(args_list)

        if with_input_tracing:
            if wrap_inputs_fn is None:
                # We control the input argument structure w.r.t. tensors
                # - a simple objwalk application should be sufficient in this simple case.
                # For more control, wrap_inputs_fn is used when this is used in NNCFNetwork
                # which is guaranteed to be the same as during the actual NNCFNetwork.forward
                args, kwargs = wrap_nncs_model_inputs_with_objwalk(args, kwargs)
            else:
                args, kwargs = wrap_inputs_fn(args, kwargs)

        return model(*args, **kwargs)

    return default_dummy_forward_fn
