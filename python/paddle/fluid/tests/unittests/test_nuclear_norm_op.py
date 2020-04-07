#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestNuclearNormOp(OpTest):
    def setUp(self):
        self.op_type = "nuclear_norm"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'X': (np.arange(9) - 4).reshape((3, 3)).astype(self.dtype),
        }
        u, s, v = np.linalg.svd(self.inputs["X"], full_matrices=1, compute_uv=1)
        self.outputs = {'Out': s.sum(), 'U': u, 'V': v}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(no_check_set=['U', 'V'], atol=1e-5)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
