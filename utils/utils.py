"""
TODO
"""

import os
import pickle
import re
import time
from datetime import datetime
from typing import Tuple, Callable
from pathlib import Path

import numpy as np
import tensorflow as tf
import sonnet as snt
import ray
import pygame
import matplotlib.pyplot as plt

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


def _get_saliency_map(input_img, gradients, convert_to_uint8: bool = True):
    assert input_img.shape == gradients.shape
    gradients = tf.abs(gradients)
    gradients = tf.math.divide(tf.subtract(gradients,
                                           tf.reduce_min(gradients)),
                               tf.subtract(tf.reduce_max(gradients),
                                           tf.reduce_min(gradients)))
    saliency_map = tf.stack([gradients, input_img, input_img], axis=-1)
    return (saliency_map if not convert_to_uint8
            else tf.cast(tf.multiply(saliency_map, 255), dtype=tf.uint8))


def _get_q_values_plot(q_values):
    plt.figure()
    plt.bar(x=range(len(q_values)), height=q_values)
    plt.show()


def visualize_policy(policy, env, num_episodes: int,
                     fps: int = 0, epsilon_greedy: float = 0.025,):
    # TODO: plot Q values
    # TODO: Jacobian matrix (sensitiviy to the different regions of the input)
    # display = pygame.display.set_mode((960, 540))
    # clock = pygame.time.Clock()

    env.reset()
    for episode in range(num_episodes):
        obs = env.reset().observation
        episode_reward = 0.0

        done = False
        while not done:
            # Rendering:
            # env_rgb = env.render(mode="rgb_array")
            env.render()
            time.sleep(1 / fps)

            # Q-values:
            obs = tf.Variable(tf.expand_dims(obs, axis=0))
            with tf.GradientTape() as tape:
                tape.watch(obs)
                q_values = policy(obs)[0]
                max_q = q_values[tf.argmax(q_values)]

            # Plotting saliency map and Q-values:
            saliency_map = _get_saliency_map(
                input_img=obs[0, :, :, -1],
                gradients=tape.gradient(max_q, obs)[0, :, :, -1],
                convert_to_uint8=True,
            )
            # _get_q_values_plot(q_values)

            # Random action:
            if np.random.uniform(low=0, high=1) < epsilon_greedy:
                action = np.random.randint(low=0,
                                           high=env.action_spec().num_values)
            # Greedy policy:
            else:
                action = tf.argmax(q_values)

            # Environment step:
            timestep_obj = env.step(action)
            obs = timestep_obj.observation

            episode_reward += timestep_obj.reward
            done = timestep_obj.last()

        print(f"Episode reward: {episode_reward}")


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

    return best_policy_path


def collect_data_from_human(env):
    data = expert_demonstration.human_play(env, num_episodes=20)

    date_and_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = f"./human_data/data_{date_and_time}.pkl"
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    count = 0
    for ep_data in data:
        count = 1 + len(ep_data["mid"])
    print(f"\nCollected data from {count} timesteps.\n")
