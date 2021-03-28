import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import acme
import tensorflow as tf
# from acme.agents.tf import dqn
import sonnet as snt
from acme.tf.networks import duelling
from acme.utils import loggers
import matplotlib.pyplot as plt
import numpy as np

from mario_env import MarioEnvironment
from utils.env_loop import EnvironmentLoop
from utils.utils import restore_module
import time
import trfl
from datetime import datetime
import pickle

import expert_demonstration
from dqn.agent import DQN as DQNAgent


def make_dqn(num_actions: int):
    return snt.Sequential([
        snt.Conv2D(32, [3, 3], [2, 2]),
        tf.nn.relu,
        snt.Conv2D(32, [3, 3], [2, 2]),
        tf.nn.relu,
        snt.Conv2D(32, [3, 3], [2, 2]),
        tf.nn.relu,
        snt.Conv2D(32, [3, 3], [2, 2]),
        tf.nn.relu,
        snt.Flatten(),
        duelling.DuellingMLP(num_actions, hidden_sizes=[512]),
    ])


def make_env():
    return MarioEnvironment(skip_frames=3,
                            img_rescale_pc=0.5,
                            stack_func=np.dstack,
                            stack_mode="all",
                            grayscale=True,
                            black_background=True,
                            in_game_score_weight=0.02,
                            movement_type="right_only",
                            world_and_level=(2, 1),
                            idle_frames_threshold=800)


def train(network=None, expert_data_path=None):
    env = make_env()
    env_spec = acme.make_environment_spec(env)

    if network is None:
        network = make_dqn(env_spec.actions.num_values)

    expert_data = None
    if expert_data_path is not None:
        with open(expert_data_path, "rb") as handle:
            expert_data = pickle.load(handle)

    agent = DQNAgent(environment_spec=env_spec,
                     network=network,
                     batch_size=32,
                     learning_rate=1e-4,
                     logger=loggers.NoOpLogger(),
                     min_replay_size=2500,
                     max_replay_size=int(1e5),
                     target_update_period=2500,
                     epsilon=tf.Variable(0.025),
                     n_step=10,
                     discount=0.9,
                     expert_data=expert_data)

    loop = EnvironmentLoop(environment=env,
                           actor=agent,
                           module2save=network)
    reward_history = loop.run(num_steps=int(1e6),
                              render=True,
                              checkpoint=True,
                              checkpoint_freq=10)

    avg_hist = [np.mean(reward_history[i:(i+50)])
                for i in range(len(reward_history) - 50)]
    plt.plot(list(range(len(avg_hist))), avg_hist)
    plt.show()

    env.close()
    return network


def eval_policy(policy, num_episodes, fps=0, epsilon_greedy=0.025):
    # TODO: plot Q values
    # TODO: Jacobian matrix (sensitiviy to the different regions of the input)
    policy = snt.Sequential([
        policy,
        lambda q: trfl.epsilon_greedy(q, epsilon=epsilon_greedy).sample(),
    ])

    env = make_env()
    env.reset()
    env.render()
    input("\nPress any key to continue.")

    for episode in range(num_episodes):
        obs = env.reset().observation
        episode_reward = 0.0

        done = False
        while not done:
            env.render()
            time.sleep(1 / fps)

            action = policy(tf.expand_dims(obs, axis=0))[0]
            timestep_obj = env.step(action)

            obs = timestep_obj.observation
            episode_reward += timestep_obj.reward
            done = timestep_obj.last()

        print(f"Episode reward: {episode_reward}")


def collect_data_from_human():
    data = expert_demonstration.human_play(make_env(), num_episodes=20)

    date_and_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = f"./human_data/data_{date_and_time}.pkl"
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    count = 0
    for ep_data in data:
        count = 1 + len(ep_data["mid"])
    print(f"\nCollected data from {count} timesteps.\n")


if __name__ == "__main__":
    # collect_data_from_human()

    policy_network = make_dqn(make_env().action_spec().num_values)
    restore_module(base_module=policy_network,
                   save_path="checkpoints/"
                             "checkpoints_2021-03-27-22-45-46/episode340_avg10-r1963_avg50-r1962_cur-r2849")

    train(policy_network, expert_data_path=None)
    eval_policy(policy_network, num_episodes=3, fps=30, epsilon_greedy=0)

    # env = make_env()
    # obs = env.reset().observation
    # while True:
    #     env.render()
    #     env.plot_obs(np.hstack([obs[:, :, i] for i in range(obs.shape[-1])]))
    #     obs = env.step(
    #         np.random.randint(low=0, high=env.action_spec().num_values)
    #     ).observation

    # env = make_env()
    # print(env.reset().observation.shape)
    # network = make_dqn(env.action_spec().num_values)
    # out = network(tf.expand_dims(env.reset().observation, axis=0))
    # print(out)
    # print(out.shape)
