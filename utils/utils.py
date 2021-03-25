"""
TODO
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
import sonnet as snt


def format_eta(eta_secs: int) -> str:
    m, s = divmod(int(eta_secs), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def save_policy(policy_network: snt.Sequential,
                input_shape: Tuple[int, int],
                save_path: str) -> None:
    @tf.function(input_signature=[tf.TensorSpec(input_shape)])
    def inference(x):
        return policy_network(x)

    to_save = snt.Module()
    to_save.inference = inference
    to_save.all_variables = list(policy_network.variables)
    tf.saved_model.save(to_save, save_path)


class LoadedModule(snt.Module):
    def __init__(self, loaded_obj):
        super().__init__()
        self._obj = loaded_obj
        self._all_variables = list(
            self._obj.signatures['serving_default'].variables
        )

    def __call__(self, x):
        return self._obj.inference(x)


def load_policy(load_path):
    loaded_container = tf.saved_model.load(load_path)
    return LoadedModule(loaded_container)


def save_module(module: snt.Module, file_prefix: str):
    checkpoint = tf.train.Checkpoint(root=module)
    return checkpoint.write(file_prefix=file_prefix)


def restore_module(base_module: snt.Module, save_path: str):
    checkpoint = tf.train.Checkpoint(root=base_module)
    return checkpoint.read(save_path=save_path)


def make_weighted_avg(weights):
    assert np.abs(np.sum(weights) - 1) < 1e-5, f"{np.sum(weights)} != 1"
    def weighted_avg(arrays):
        array_sum = arrays[0] * weights[0]
        for i in range(1, len(arrays)):
            array_sum = array_sum + arrays[i] * weights[i]
        return array_sum / np.sum(weights)

    return weighted_avg
