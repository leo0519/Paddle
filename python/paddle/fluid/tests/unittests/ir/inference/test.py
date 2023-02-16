import unittest

import hypothesis.strategies as st
from auto_scan_test import IgnoreReasons, PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig
import paddle.inference as paddle_infer
from typing import Dict
import numpy as np
import os

class TestMapMatmulToMulPass(PassAutoScanTest):
    r"""
    x_var    y_var(persistable)
      \       /
      matmul_v2
    """

    def sample_predictor_configs(self, program_config):
        # TRT
        config = self.create_trt_inference_config()
        if self.precision == 'FP32':
            precision_mode = paddle_infer.PrecisionType.Float32
        else:
            precision_mode = paddle_infer.PrecisionType.Half
        config.enable_tensorrt_engine(
             max_batch_size=10,
             workspace_size=4194304,
             min_subgraph_size=0,
             precision_mode=precision_mode,
             use_static=False,
            use_calib_mode=False)

        config.set_trt_dynamic_shape_info(
            {
                "matmul_x": [1, 64, self.k_size],
                "matmul_y": [self.k_size, 1248],
            },
            {
                "matmul_x": [64, 64, self.k_size],
                "matmul_y": [self.k_size, 1248],
            },
            {
                "matmul_x": [32, 64, self.k_size],
                "matmul_y": [self.k_size, 1248],
            },
        )
        if self.precision == 'FP32':
            tol = 1e-5
        else:
            tol = 1e-3
        yield config, ["mul", ], (tol, tol)

    def sample_program_config(self, draw):
        # 1. Generate shape and attr of matmul
        x_shape = [1, 64, self.k_size]
        y_shape = [self.k_size, 1248]
        alpha = 1.0
        transpose_X = False
        transpose_Y = False

        matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["matmul_x"], "Y": ["matmul_y"]},
            outputs={"Out": ["matmul_out"]},
            alpha=alpha,
            trans_x=transpose_X,
            trans_y=transpose_Y,
        )

        ops = [
            matmul_op,
        ]
        weights = {
            "matmul_y": TensorConfig(shape=y_shape),
        }
        inputs = {
            "matmul_x": TensorConfig(shape=x_shape),
        }
        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs=inputs,
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.precision = os.environ.get('PRECISION', default='FP32')
        self.k_size = int(os.environ.get('K_SIZE', default='312'))
        self.run_and_statis(
            quant=False,
            max_examples=1,
            min_success_num=1,
            passes=["trt_map_matmul_v2_to_mul_pass"],
        )
    def assert_tensors_near(self, atol: float, rtol: float,
                            tensor: Dict[str, np.array],
                            baseline: Dict[str, np.array]):
        for key, arr in tensor.items():
            self.assertEqual(
                baseline[key].shape, arr.shape,
                'The output shapes are not equal, the baseline shape is ' +
                str(baseline[key].shape) + ', but got ' + str(arr.shape))
            abs_err = np.abs(baseline[key] - arr)
            rel_err = abs_err / baseline[key]
            max_idx = np.unravel_index(np.argmax(rel_err, axis=None), rel_err.shape)
            print("The index of the max rel_err is", max_idx)
            print("The corresponding element in baseline is", baseline[key][max_idx])
            print("The corresponding element in TRT result is", arr[max_idx])
            np.testing.assert_allclose(baseline[key], arr, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
