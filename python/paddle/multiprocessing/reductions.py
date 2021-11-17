# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

# TODO: check the hooks of tensor
# TODO: check serializing named tensor
# TODO: check influence on autograd
import os
import math
import threading
import multiprocessing
from multiprocessing.util import register_after_fork
from multiprocessing.reduction import ForkingPickler


# TODO: support cuda EVENT ipc
def rebuild_event():
    pass


def reduce_event():
    pass


def rebuild_tensor(cls, lodtensor, metadata):
    size, stop_gradient = metadata
    # TODO: the place the generate by GetCurrentTracer, show be set by lod tensor place
    # https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/pybind/imperative.cc#L265
    # t = paddle.fluid.core.VarBase(lodtensor)

    t = paddle.fluid.core.VarBase()
    t.value().get_tensor()._share_data_with(lodtensor)

    if cls == paddle.fluid.framework.ParamBase:
        # TODO: Add ParamBase init with lodtensor
        raise NotImplementedError(
            "paddle.fluid.framework.ParamBase init with VarBase")
        #t = paddle.nn.parameter.Parameter(t, trainable=not stop_gradient)
    else:
        t.stop_gradient = stop_gradient
    return t


def reduce_tensor(tensor):
    lodtensor = tensor.value().get_tensor()

    if not tensor.stop_gradient and not tensor.is_leaf:
        raise RuntimeError(
            "Refusing to serialize non-leaf tensor which not stop_gradient, you can deteach it!"
        )
    # TODO: add serializing name and  hooks check
    if not tensor.place.is_cpu_place():
        raise ValueError("Not support cuda, xpu tensor for now!")
    # Don't serialize hooks
    metadata = (tensor.size, tensor.stop_gradient)
    return (rebuild_tensor, (type(tensor), lodtensor, metadata))


def rebuild_lodtensor_filename(cls, ipc_name, size, type_idx, dims, lod):
    lodtensor = cls._new_shared_filename((ipc_name, size, type_idx, dims, lod))
    return lodtensor


def rebuild_lodtensor_empty(cls):
    #TODO: check if tensor initialized
    #TODO: handle the dtype of empty tensor
    return cls()


def reduce_lodtensor(lodtensor):
    if not lodtensor._place().is_cpu_place():
        raise RuntimeError("We only support pass cpu lodtensor for now!")
    # Default use share filename stratege
    metadata = lodtensor._share_filename_(
    )  # ipc_name, size, type_idx, dims, lod
    rebuild = rebuild_lodtensor_filename
    # TODO, shared incref for lodtensor
    for dim in lodtensor.shape():
        if dim == 0:
            # Empty tensors have nothing be mmapped.
            return (rebuild_lod_empty, (type(lodtensor), ))
    # TODO: support file_discriptor stratege

    return (rebuild, (type(lodtensor), ) + metadata)


def init_reductions():
    # TODO: support cuda event
    #ForkingPickler.register(paddle.cuda.Event, reduce_event)

    ForkingPickler.register(paddle.Tensor, reduce_tensor)
    ForkingPickler.register(paddle.fluid.core.VarBase, reduce_tensor)
    # TODO: support Variable ParamBase
    # ForkingPickler.register(paddle.fluid.core.Variable, reduce_variable)
    ForkingPickler.register(paddle.fluid.core.LoDTensor, reduce_lodtensor)
