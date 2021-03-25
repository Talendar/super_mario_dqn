""" A simple agent-environment training loop."""

import operator
import time
from typing import Optional

from acme import core
# Internal imports.
from acme.utils import counting
from acme.utils import loggers

import dm_env
from dm_env import specs
import numpy as np
import tree
from collections import deque
from utils.utils import format_eta, save_module
from datetime import datetime
import os
import sonnet as snt


class EnvironmentLoop:
    def __init__(
            self,
            environment: dm_env.Environment,
            actor: core.Actor,
            module2save: Optional[snt.Module] = None):
        # Internalize agent and environment.
        self._environment = environment
        self._actor = actor
        self._counter = counting.Counter()
        self._module2save = module2save

    def run_episode(self,
                    render: bool,
                    update_actor: bool) -> loggers.LoggingData:
        """Run one episode.

        Each episode is a loop which interacts first with the environment to get
        an observation and then give that observation to the agent in order to
        retrieve an action.

        Returns:
          An instance of `loggers.LoggingData`.
        """
        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0

        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode.
        episode_return = tree.map_structure(_generate_zeros_from_spec,
                                            self._environment.reward_spec())
        timestep = self._environment.reset()

        # Make the first observation.
        self._actor.observe_first(timestep)

        # Run an episode.
        while not timestep.last():
            if render:
                self._environment.render()

            # Generate an action from the agent's policy and step the env.
            action = self._actor.select_action(timestep.observation)
            timestep = self._environment.step(action)

            # Have the agent observe the timestep and let the actor update
            # itself.
            self._actor.observe(action, next_timestep=timestep)
            if update_actor:
                self._actor.update()

            # Book-keeping.
            episode_steps += 1

            # Equivalent to: episode_return += timestep.reward
            tree.map_structure(operator.iadd, episode_return, timestep.reward)

        # Record counts.
        counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
        }
        result.update(counts)
        return result

    def run(self,
            num_steps: int,
            render: bool = False,
            update_actor: bool = True,
            checkpoint: bool = True,
            checkpoint_path: Optional[str] = None,
            checkpoint_freq: int = 100):
        # Checkpoints preparation:
        if checkpoint:
            # Assertion:
            if self._module2save is None:
                raise ValueError("In order to make checkpoints, you must "
                                 "specify a Sonnet module to be saved!")

            # Choose path if one hasn't been chosen yet:
            if checkpoint_path is None:
                date_and_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
                checkpoint_path = f"./checkpoints/checkpoints_{date_and_time}"

            # Initial checkpoint:
            save_module(module=self._module2save,
                        file_prefix=os.path.join(checkpoint_path, "episode0"))

        # Counters:
        steps_per_sec = deque(maxlen=25)
        episodes_rewards = []
        episode_count = step_count = 0

        # Main loop:
        while step_count < num_steps:
            # Running episode:
            result = self.run_episode(render=render, update_actor=update_actor)

            # Updating counters:
            episode_count += 1
            step_count += result['episode_length']
            steps_per_sec.append(result["steps_per_second"])
            episodes_rewards.append(result["episode_return"])

            # Stats:
            eta = format_eta(
                max(0, (num_steps - step_count) // np.mean(steps_per_sec))
            )
            avg_reward10 = np.mean(episodes_rewards
                                   if len(episodes_rewards) < 10
                                   else episodes_rewards[-10:])
            avg_reward50 = np.mean(episodes_rewards
                                   if len(episodes_rewards) < 50
                                   else episodes_rewards[-50:])

            # Verbose:
            print(f"[{step_count/num_steps:.2%}][E: {episode_count}] "
                  f"Steps: {step_count}/{num_steps}  |  "
                  f"Steps per sec: {result['steps_per_second']:.2f}  |  "
                  f"ETA: {eta}\n"
                  f"Episode reward: {result['episode_return']:.2f}  |  "
                  f"Avg. reward (10 ep.): {avg_reward10:.2f}  |  "
                  f"Avg. reward (50 ep.): {avg_reward50:.2f}")

            # Checkpoint:
            if checkpoint and (episode_count % checkpoint_freq) == 0:
                save_path = os.path.join(
                    checkpoint_path,
                    f"episode{episode_count}_"
                    f"avg10-r{int(avg_reward10)}_"
                    f"avg50-r{int(avg_reward50)}_"
                    f"cur-r{int(result['episode_return'])}"
                )
                save_module(module=self._module2save, file_prefix=save_path)
                print(f"Checkpoint saved to: {save_path}")
            print()

        # Final checkpoint:
        if checkpoint and step_count > 0:
            save_path = os.path.join(
                checkpoint_path,
                f"episode{episode_count}_final_"
                f"avg10-r{int(avg_reward10)}_"
                f"avg50-r{int(avg_reward50)}_"
                f"cur-r{int(result['episode_return'])}"
            )
            save_module(module=self._module2save, file_prefix=save_path)

        return episodes_rewards


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)
