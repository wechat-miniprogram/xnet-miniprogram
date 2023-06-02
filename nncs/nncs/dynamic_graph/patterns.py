"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from nncs.dynamic_graph.graph import NNCSNodeExpression as N
from nncs.dynamic_graph.version_agnostic_op_names import VersionAgnosticNames

LINEAR_OPS = (
    N("linear")
    | N("conv2d")
    | N("conv_transpose2d")
    | N("conv3d")
    | N("conv_transpose3d")
    | N("conv1d")
    | N("addmm")
)

RELU = N(VersionAgnosticNames.RELU) | N("hardtanh") | N("clamp")

BN = N("batch_norm") | N("batch_norm3d")

POOLING = (
    N("adaptive_avg_pool2d")
    | N("adaptive_avg_pool3d")
    | N("avg_pool2d")
    | N("avg_pool3d")
)

# NON_RELU_ACTIVATIONS = N('elu') | N('elu_') | N('prelu') | N('gelu') #| N('sigmoid')

ACTIVATIONS = RELU  # | NON_RELU_ACTIVATIONS

ANY_BN_ACT_COMBO = BN + ACTIVATIONS | ACTIVATIONS + BN | BN | ACTIVATIONS

SINGLE_OPS = ACTIVATIONS | POOLING | N("mean") | N("layer_norm")

ARITHMETIC = N("__iadd__") | N("__add__") | N("__mul__") | N("__rmul__")

ELTWISE_UNIFORM_OPS = BN | RELU | ACTIVATIONS

MATMUL = N("matmul") | N("bmm")

LOG_SOFTMAX = N("softmax") + N("log")
