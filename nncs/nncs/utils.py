from collections import OrderedDict
from typing import Dict, Callable, Any, Mapping, Sequence, Set, List
from contextlib import contextmanager

import torch

string_types = (str, bytes)


def iteritems(mapping):
    return getattr(mapping, "iteritems", mapping.items)()


def _parent_name(target):
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


def get_module(model, submodule_key):
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def get_node_name(module, module_name, prefix):
    return "{prefix}/{cls}[{name}]".format(
        prefix=prefix, cls=module.__class__.__name__, name=module_name
    )


def get_all_modules(model, prefix=None):
    found = OrderedDict()
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        found[full_node_name] = module
        sub_found = get_all_modules(module, prefix=full_node_name)
        if sub_found:
            found.update(sub_found)
    return found


def get_module_by_node_name(
    model: torch.nn.Module, node_scope_str: str, prefix=None
) -> torch.nn.Module:
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        if full_node_name == node_scope_str:
            return module
        sub_result = get_module_by_node_name(module, node_scope_str, full_node_name)
        if sub_result is not None:
            return sub_result
    return None


def set_module_by_node_name(model, node_name, module_to_set, prefix=None):
    if prefix is None:
        prefix = model.__class__.__name__

    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        if full_node_name == node_name:
            # pylint: disable=protected-access
            model._modules[name] = module_to_set
        set_module_by_node_name(module, node_name, module_to_set, full_node_name)


def get_module_by_nodename(base_module, node_name):
    # pylint: disable=unused-variable
    node_idx, node_remain = node_name.split(" ")
    parts = node_remain.split("/")

    # pylint: disable=unused-variable
    base_module_name = parts[0]
    # pylint: disable=unused-variable
    op_type = parts[-1]

    m = base_module
    for part in parts[1:-1]:
        lindex = part.index("[")
        rindex = part.rindex("]")
        name = part[lindex + 1 : rindex]
        m = getattr(m, name)

    return m


def set_module_by_nodename(base_module, node_name, new_module):
    # pylint: disable=unused-variable
    node_idx, node_remain = node_name.split(" ")
    parts = node_remain.split("/")

    # pylint: disable=unused-variable
    base_module_name = parts[0]
    # pylint: disable=unused-variable
    op_type = parts[-1]

    if len(parts) - 2 == 1:
        part = parts[1]
        lindex = part.index("[")
        rindex = part.rindex("]")
        name = part[lindex + 1 : rindex]
        setattr(base_module, name, new_module)
        return

    m = base_module
    for part in parts[1:-2]:
        lindex = part.index("[")
        rindex = part.rindex("]")
        name = part[lindex + 1 : rindex]
        m = getattr(m, name)

    part = parts[-2]
    lindex = part.index("[")
    rindex = part.rindex("]")
    name = part[lindex + 1 : rindex]
    setattr(m, name, new_module)


def get_mname_by_nname(prefix, node_name):
    # pylint: disable=unused-variable
    node_idx, node_remain = node_name.split(" ", 1)
    parts = node_remain.split("/")

    m = []
    for part in parts[1:-1]:
        lindex = part.index("[")
        rindex = part.rindex("]")
        name = part[lindex + 1 : rindex]
        m.append(name)

    return ".".join(m)


def parse_node_name(name):
    slash_pos = -1
    nbrackets = 0
    for i, ch in enumerate(reversed(name)):
        if ch == "]":
            nbrackets += 1
        elif ch == "[":
            nbrackets -= 1
        elif ch == "/" and nbrackets == 0:
            slash_pos = len(name) - i - 1
            break

    prefix = None if slash_pos < 0 else name[:slash_pos]

    last_name = name[slash_pos + 1 :]
    open_bracket_pos = last_name.find("[")
    if open_bracket_pos < 0:
        return prefix, last_name, None
    return prefix, last_name[:open_bracket_pos], last_name[open_bracket_pos + 1 : -1]


def is_tensor(obj):
    return isinstance(obj, torch.Tensor)


def maybe_get_iterator(obj):
    it = None
    # pylint:disable=isinstance-second-argument-not-valid-type
    if isinstance(obj, Mapping):
        it = iteritems
        # pylint:disable=isinstance-second-argument-not-valid-type
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types):
        it = enumerate
    return it


def objwalk(obj, unary_predicate: Callable[[Any], bool], apply_fn: Callable, memo=None):
    if memo is None:
        memo = set()

    is_tuple = isinstance(obj, tuple)
    if is_tuple:
        obj = list(obj)

    iterator = maybe_get_iterator(obj)

    if iterator is not None:
        if id(obj) not in memo:
            memo.add(id(obj))
            indices_to_apply_fn_to = set()
            indices_vs_tuples_to_assign = {}  # type: Dict[Any, list]
            for idx, value in iterator(obj):
                next_level_it = maybe_get_iterator(value)
                if next_level_it is None:
                    if unary_predicate(value):
                        indices_to_apply_fn_to.add(idx)
                else:
                    if isinstance(value, tuple):
                        processed_tuple = objwalk(
                            value, unary_predicate, apply_fn, memo
                        )
                        indices_vs_tuples_to_assign[idx] = processed_tuple
                    else:
                        objwalk(value, unary_predicate, apply_fn)
            for idx in indices_to_apply_fn_to:
                obj[idx] = apply_fn(obj[idx])
            for idx, tpl in indices_vs_tuples_to_assign.items():
                obj[idx] = tuple(tpl)

            memo.remove(id(obj))
    else:
        if unary_predicate(obj):
            return apply_fn(obj)

    if is_tuple:
        return tuple(obj)

    return obj


def should_consider_scope(
    scope_str: str, target_scopes: List[str], ignored_scopes: List[str]
):
    return (
        target_scopes is None or in_scope_list(scope_str, target_scopes)
    ) and not in_scope_list(scope_str, ignored_scopes)


@contextmanager
def training_mode_switcher(model: torch.nn.Module, is_training: bool = True):
    is_original_mode_training = model.training
    model.train(is_training)
    try:
        yield
    finally:
        model.train(is_original_mode_training)
