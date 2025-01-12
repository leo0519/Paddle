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

import paddle.fluid.core as core
import paddle.fluid.eager.eager_tensor_patch_methods as eager_tensor_patch_methods
import paddle
import numpy as np
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.data_feeder import convert_dtype
import unittest


class EagerScaleTestCase(unittest.TestCase):
    def test_scale_base(self):
        with _test_eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            tensor = paddle.to_tensor(arr, 'float32', core.CPUPlace())
            print(tensor)
            tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            for i in range(0, 100):
                tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            print(tensor)
            self.assertEqual(tensor.shape, [4, 16, 16, 32])
            self.assertEqual(tensor.stop_gradient, True)

    def test_retain_grad_and_run_backward(self):
        with _test_eager_guard():
            paddle.set_device("cpu")

            input_data = np.ones([4, 16, 16, 32]).astype('float32')
            data_eager = paddle.to_tensor(input_data, 'float32',
                                          core.CPUPlace(), False)

            grad_data = np.ones([4, 16, 16, 32]).astype('float32')
            grad_eager = paddle.to_tensor(grad_data, 'float32', core.CPUPlace())

            core.eager.retain_grad_for_tensor(data_eager)

            out_eager = core.eager.scale(data_eager, 1.0, 0.9, True, True)
            self.assertFalse(data_eager.grad._is_initialized())
            core.eager.run_backward([out_eager], [grad_eager], False)
            self.assertTrue(data_eager.grad._is_initialized())
            self.assertTrue(np.array_equal(data_eager.grad.numpy(), input_data))


class EagerDtypeTestCase(unittest.TestCase):
    def check_to_tesnsor_and_numpy(self, dtype, proto_dtype):
        with _test_eager_guard():
            arr = np.random.random([4, 16, 16, 32]).astype(dtype)
            tensor = paddle.to_tensor(arr, dtype)
            self.assertEqual(tensor.dtype, proto_dtype)
            self.assertTrue(np.array_equal(arr, tensor.numpy()))

    def test_dtype_base(self):
        print("Test_dtype")
        self.check_to_tesnsor_and_numpy('bool', core.VarDesc.VarType.BOOL)
        self.check_to_tesnsor_and_numpy('int8', core.VarDesc.VarType.INT8)
        self.check_to_tesnsor_and_numpy('uint8', core.VarDesc.VarType.UINT8)
        self.check_to_tesnsor_and_numpy('int16', core.VarDesc.VarType.INT16)
        self.check_to_tesnsor_and_numpy('int32', core.VarDesc.VarType.INT32)
        self.check_to_tesnsor_and_numpy('int64', core.VarDesc.VarType.INT64)
        self.check_to_tesnsor_and_numpy('float16', core.VarDesc.VarType.FP16)
        self.check_to_tesnsor_and_numpy('float32', core.VarDesc.VarType.FP32)
        self.check_to_tesnsor_and_numpy('float64', core.VarDesc.VarType.FP64)
        self.check_to_tesnsor_and_numpy('complex64',
                                        core.VarDesc.VarType.COMPLEX64)
        self.check_to_tesnsor_and_numpy('complex128',
                                        core.VarDesc.VarType.COMPLEX128)


class EagerTensorPropertiesTestCase(unittest.TestCase):
    def test_properties(self):
        print("Test_properties")
        with _test_eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            tensor = paddle.to_tensor(arr, core.VarDesc.VarType.FP32,
                                      core.CPUPlace())
            self.assertEqual(tensor.shape, [4, 16, 16, 32])
            tensor.name = 'tensor_name_test'
            self.assertEqual(tensor.name, 'tensor_name_test')
            self.assertEqual(tensor.persistable, False)
            tensor.persistable = True
            self.assertEqual(tensor.persistable, True)
            tensor.persistable = False
            self.assertEqual(tensor.persistable, False)
            self.assertTrue(tensor.place.is_cpu_place())
            self.assertEqual(tensor._place_str, 'CPUPlace')
            self.assertEqual(tensor.stop_gradient, True)
            tensor.stop_gradient = False
            self.assertEqual(tensor.stop_gradient, False)
            tensor.stop_gradient = True
            self.assertEqual(tensor.stop_gradient, True)

    def test_global_properties(self):
        print("Test_global_properties")
        self.assertFalse(core._in_eager_mode())
        with _test_eager_guard():
            self.assertTrue(core._in_eager_mode())
        self.assertFalse(core._in_eager_mode())

    def test_place_guard(self):
        core._enable_eager_mode()
        if core.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
            with paddle.fluid.framework._dygraph_place_guard(core.CPUPlace()):
                self.assertTrue(core.eager._get_expected_place().is_cpu_place())
        else:
            paddle.set_device("cpu")
            with paddle.fluid.framework._dygraph_place_guard(core.CPUPlace()):
                self.assertTrue(core.eager._get_expected_place().is_cpu_place())
        core._disable_eager_mode()


if __name__ == "__main__":
    unittest.main()
