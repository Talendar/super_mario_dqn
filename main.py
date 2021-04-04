import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import acme
import sonnet as snt
from acme.tf.networks import duelling
from acme.utils import loggers
import matplotlib.pyplot as plt
import numpy as np

from mario_env import MarioEnvironment
from utils.env_loop import EnvironmentLoop
from utils.utils import restore_module
from utils.utils import collect_data_from_human
import pickle

from dqn.agent import DQN as DQNAgent
from utils import visualize_policy
from utils import find_best_policy


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


def make_env(colorful_rendering: bool = False):
    return MarioEnvironment(skip_frames=3,
                            img_rescale_pc=0.5,
                            stack_func=np.dstack,
                            stack_mode="all",
                            grayscale=True,
                            black_background=True,
                            in_game_score_weight=0.025,
                            movement_type="right_only",
                            world_and_level=(4, 3),
                            idle_frames_threshold=1250,
                            colorful_rendering=colorful_rendering)


def train(network=None, expert_data_path=None):
    env = make_env()
    env_spec = acme.make_environment_spec(env)

    if network is None:
        network = make_dqn(env_spec.actions.num_values)

    expert_data = None
    if expert_data_path is not None:
        with open(expert_data_path, "rb") as handle:
            expert_data = pickle.load(handle)
        num_timesteps = np.sum([1 + len(ep["mid"]) for ep in expert_data])
        print(f"Using expert data from {expert_data_path}. "
              f"Episodes: {len(expert_data)}. Timesteps: {num_timesteps}.")

    agent = DQNAgent(environment_spec=env_spec,
                     network=network,
                     batch_size=32,
                     learning_rate=6.25e-5,
                     logger=loggers.NoOpLogger(),
                     min_replay_size=1000,
                     max_replay_size=int(1e5),
                     target_update_period=2500,
                     epsilon=tf.Variable(0.032),
                     n_step=20,
                     discount=0.97,
                     expert_data=expert_data)

    loop = EnvironmentLoop(environment=env,
                           actor=agent,
                           module2save=network)
    reward_history = loop.run(num_steps=int(1e6),
                              render=True,
                              checkpoint=True,
                              checkpoint_freq=15)

    avg_hist = [np.mean(reward_history[i:(i+50)])
                for i in range(len(reward_history) - 50)]
    plt.plot(list(range(len(avg_hist))), avg_hist)
    plt.show()

    env.close()
    return network


if __name__ == "__main__":
    # collect_data_from_human(env=make_env(colorful_rendering=True),
    #                         num_episodes=20)

    policy_path, policies_rewards = find_best_policy(
        folder_path="checkpoints/checkpoints_2021-04-04-00-12-22/",
        make_env=make_env,
        make_dqn=make_dqn,
    )
    print()
    for fn, reward in policies_rewards.items():
        print(f"[Reward: {reward}] {fn}")

    # policy_path = "checkpoints/checkpoints_2021-04-03-23-25-07/episode1300_avg10-r563_avg50-r546_cur-r564"

    policy_network = make_dqn(make_env().action_spec().num_values)
    restore_module(base_module=policy_network, save_path=policy_path)
    print(f"\nUsing policy checkpoint from: {policy_path}")

    # train(policy_network, expert_data_path=None)

    input("\nPress [ENTER] to continue.")
    visualize_policy(policy_network, env=make_env(colorful_rendering=True),
                     num_episodes=1, fps=120, epsilon_greedy=0,
                     plot_extras=True, save_video=False)

    # DEBUG
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
