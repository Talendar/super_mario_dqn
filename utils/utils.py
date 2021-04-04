"""
TODO
"""

import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import ray
import sonnet as snt
import tensorflow as tf

import expert_demonstration


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


def find_best_policy(folder_path: str,
                     make_env: Callable,
                     make_dqn: Callable):
    ray.init(ignore_reinit_error=True)

    @ray.remote
    def _eval_policy_ray(policy_path):
        env = make_env()
        policy = make_dqn(env.action_spec().num_values)
        restore_module(base_module=policy, save_path=policy_path)

        obs = env.reset().observation
        policy_reward = 0
        done = False
        while not done:
            q_values = policy(tf.expand_dims(obs, axis=0))[0]
            action = tf.argmax(q_values)

            timestep_obj = env.step(action)
            obs = timestep_obj.observation

            policy_reward += timestep_obj.reward
            done = timestep_obj.last()

        return policy_reward

    # Getting files names:
    files = []
    for i, fn in enumerate(sorted(Path(folder_path).iterdir(),
                                  key=os.path.getmtime)):
        if i % 2 == 0:
            files.append(re.search("^[^.]*", str(fn))[0])

    # Searching:
    futures = [_eval_policy_ray.remote(fn) for fn in files]
    rewards = ray.get(futures)

    best_policy_path = files[np.argmax(rewards)]
    best_policy_reward = np.max(rewards)

    print(f"Best policy found at: {best_policy_path}")
    print(f"Best policy total reward: {best_policy_reward}")

    policies_rewards = {fn: reward for fn, reward in zip(files, rewards)}
    policies_rewards = {k: v for k, v in sorted(policies_rewards.items(),
                                                key=lambda item: item[1])}
    return best_policy_path, policies_rewards


def collect_data_from_human(env, num_episodes=10):
    data = expert_demonstration.human_play(env, num_episodes=num_episodes)

    date_and_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = f"./human_data/data_{date_and_time}.pkl"
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    count = 0
    for ep_data in data:
        count = 1 + len(ep_data["mid"])
    print(f"\nCollected data from {count} timesteps.\n")
