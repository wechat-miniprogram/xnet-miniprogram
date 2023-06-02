from distutils.version import LooseVersion

import torch
if LooseVersion(torch.__version__) < LooseVersion('1.10.0'):
    assert(False), "fx must use torch version larger than 1.10.0"
