import warnings

from torch.nn import DataParallel

from nncs.debug import is_debug
from nncs.dynamic_graph.context import get_current_context, OperatorInput
from nncs.dynamic_graph.graph import NNCSNode
from nncs.dynamic_graph.trace_tensor import trace_tensors
from nncs.layers import ITERATION_MODULES

_IGNORED_SCOPES = []


def _warn_data_parallel():
    if getattr(_warn_data_parallel, "warned_once", False):
        return
    _warn_data_parallel.warned_once = True
    warnings.warn(
        "You are using DataParallel, which may cause significant performance issues with dynamic graph "
        "building. Consider using distributed training (DistributedDataParallel) instead"
    )


def ignore_scope(cls):
    if cls not in _IGNORED_SCOPES:
        _IGNORED_SCOPES.append(cls)
    return cls


def wrap_operator(operator, operator_info: "PatchedOperatorInfo"):
    # do not wrap function twice
    _orig_op = getattr(operator, "_original_op", None)
    if _orig_op is not None:
        raise Exception("Operator: {} is already wrapped".format(_orig_op.__name__))

    def wrapped(*args, **kwargs):

        ctx = get_current_context()
        if not ctx or getattr(ctx, "in_operator", False):  # or not ctx.is_tracing:
            result = operator(*args, **kwargs)
            return result

        ctx.in_operator = True

        if not ctx.is_tracing:
            ia_op_exec_context = ctx.get_caller_context(operator_info.name)
            ctx.register_operator_call(
                ia_op_exec_context.operator_name, ia_op_exec_context.scope_in_model
            )
            op_input = OperatorInput(list(args), kwargs)
            processed_input = ctx.execute_pre_hooks(ia_op_exec_context, op_input)
            args = tuple(processed_input.op_args)
            kwargs = processed_input.op_kwargs
            result = operator(*args, **kwargs)
            node = NNCSNode(-1, None)
            result = trace_tensors(result, node)
            result = ctx.execute_post_hooks(ia_op_exec_context, result)
            ctx.in_operator = False
            return result

        if operator_info.custom_trace_fn is not None:
            result = operator_info.custom_trace_fn(operator, *args, **kwargs)
        else:
            ia_op_exec_context = ctx.get_caller_context(operator_info.name)

            ctx.register_operator_call(
                ia_op_exec_context.operator_name, ia_op_exec_context.scope_in_model
            )

            op_input = OperatorInput(list(args), kwargs)

            processed_input = ctx.execute_pre_hooks(ia_op_exec_context, op_input)

            node = ctx.find_operator_node(processed_input, ia_op_exec_context)

            if is_debug():
                ctx.register_node_call(ctx.graph.get_node_key_by_id(node.node_id))

            args = tuple(processed_input.op_args)
            kwargs = processed_input.op_kwargs

            if ia_op_exec_context.operator_name not in ["__setitem__"]:
                result = operator(*args, **kwargs)
                result = trace_tensors(result, node)
                result = ctx.execute_post_hooks(ia_op_exec_context, result)
            else:
                operator(*args, **kwargs)
                result = args[0]
                result = trace_tensors(result, node)
                if ia_op_exec_context in ctx._post_hooks:
                    args[0][:] = ctx.execute_post_hooks(ia_op_exec_context, result)
                else:
                    pass

        ctx.in_operator = False
        return result

    # pylint: disable=protected-access
    wrapped._original_op = operator
    return wrapped


def wrap_module_call(module_call):
    def wrapped(self, *args, **kwargs):
        ctx = get_current_context()
        if not ctx or self.__class__ in _IGNORED_SCOPES:
            if isinstance(self, DataParallel):
                _warn_data_parallel()
            return module_call(self, *args, **kwargs)
        ctx.push_scope(self)
        retval = module_call(self, *args, **kwargs)
        if type(self).__name__ in ITERATION_MODULES.registry_dict.keys():
            ctx.reset_operator_call_count_in_scope(ctx.scope)
        ctx.pop_scope()
        return retval

    return wrapped
