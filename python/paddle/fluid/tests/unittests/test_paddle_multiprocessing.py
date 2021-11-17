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

import unittest
import paddle
import paddle.multiprocessing as mp
paddle.set_device("cpu")


def fill_tensor(queue, event):
    paddle.set_device("cpu")
    data = queue.get()
    data[0][:] = 5
    data[1][:] = 5
    event.set()


def send_tensor(queue, event, device, dtype):
    tensor = paddle.ones([5, 5], dtype=dtype)
    tensor = tensor.cpu()
    queue.put(tensor)
    queue.put(tensor)
    event.wait()


class TestMultiprocessingBase(unittest.TestCase):
    def get_tensor(self, device="cpu"):
        self.device = device.lower()
        place = None
        if self.device == "cpu":
            place = paddle.CPUPlace()
        elif self.device == "gpu":
            place = paddle.CUDAPlace()
        tensor = paddle.zeros([5, 5], dtype="float32")
        tensor = tensor.cpu()
        return tensor

    def _test_sharing(self, ctx=mp, device='cpu', dtype="float32", repeat=1):
        def test_fill():
            x = self.get_tensor()
            y = x[:, 1]
            data = [x, y]

            queue = ctx.Queue()
            event = ctx.Event()
            queue.put(data)

            process = ctx.Process(target=fill_tensor, args=(queue, event))
            process.daemon = True
            process.start()

            event.wait(30)

            self.assertTrue(event.is_set())
            self.assertTrue(data[0].equal(5).all())
            self.assertTrue(data[1].equal(5).all())

            process.join(1)
            self.assertFalse(process.is_alive())

        def test_receive():
            queue = ctx.Queue()
            event = ctx.Event()

            process = ctx.Process(
                target=send_tensor, args=(queue, event, device, dtype))
            process.daemon = True
            process.start()

            t1 = queue.get()
            t2 = queue.get()
            self.assertTrue(t1.equal(1).all())
            del t1, t2

            event.set()
            process.join(1)
            self.assertFalse(process.is_alive())

        for _ in range(repeat):
            test_fill()
            test_receive()


class TestMultiprocessingCpu(TestMultiprocessingBase):
    def test_pass_tensor(self):
        self._test_sharing()

    def test_pass_varbase(self):
        # TODO: check stop gradient
        pass

    def test_pass_parambase(self):
        pass

    def test_pass_lodtensor(self):
        # TODO: set lod, and check lod info
        pass

    def test_pass_variable(self):
        pass

    def test_pass_empty(self):
        pass


# class TestMultiprocessingGpu(TestMultiprocessingBase):
#     def test_pass_tensor(self):
#         pass
# 
#     def test_pass_varbase(self):
#         pass
# 
#     def test_pass_parambase(self):
#         pass
# 
#     def test_pass_lodtensor(self):
#         pass
# 
#     def test_pass_variable(self):
#         pass

if __name__ == "__main__":
    unittest.main()
