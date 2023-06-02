from .common.utils.backend import __nncs_backend__

if __nncs_backend__ == 'Torch':
    from .dynamic_graph.patch_pytorch import patch_torch_operators
    patch_torch_operators()
