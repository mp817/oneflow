"""
Copyright 2020 The OneFlow Authors. All rights reserved.

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
from collections import OrderedDict

import unittest
import numpy as np
import oneflow as flow
import oneflow.typing as tp
import tensorflow as tf
import random
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, target_dtype):
    assert device_type in ["gpu", "cpu"]
    assert target_dtype in ["int32", "int64"]
    flow_data_type = type_name_to_flow_type[target_dtype]

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow_data_type)

    targets = np.random.randint(4, size=3).astype(type_name_to_np_type[target_dtype])
    predictions = np.random.rand(3, 4).astype("float32")
    k = random.choice([1, 2, 3, 4])

    @flow.global_function(function_config=func_config)
    def IntopkJob(
        targets: tp.Numpy.Placeholder((3,), dtype=flow_data_type),
        predictions: tp.Numpy.Placeholder((3, 4), dtype=flow.float32),
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.in_top_k(targets, predictions, k=k)

    # OneFlow
    of_out = IntopkJob(targets, predictions,).get().numpy()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        predictions = tf.Variable(predictions)
        targets = tf.Variable(targets)
        tf_out = tf.math.in_top_k(targets, predictions, k=k)
    assert np.allclose(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["target_dtype"] = ["int32", "int64"]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestInTopk(flow.unittest.TestCase):
    def test_in_top_K(test_case):
        for arg in gen_arg_list():
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()