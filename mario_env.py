import dm_env
import gym_super_mario_bros
import matplotlib
import numpy as np
from dm_env import specs
from dm_env import TimeStep
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from skimage.transform import rescale

from typing import Optional, Tuple, Callable, List

RGB2GRAY_COEFFICIENTS = np.array([0.3, 0.3, 0.4], dtype=np.float32)
MOVEMENTS_TYPES = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "simple_with_down": SIMPLE_MOVEMENT + ["down"],
    "complex": COMPLEX_MOVEMENT,
}


class MarioEnvironment(dm_env.Environment):
    def __init__(self,
                 skip_frames: int = 3,
                 img_rescale_pc: float = 0.4,
                 stack_func: Optional[
                     Callable[[List[np.ndarray]], np.ndarray]] = np.hstack,
                 stack_mode: str = "all",
                 grayscale: bool = True,
                 black_background: bool = True,
                 in_game_score_weight: float = 0.01,
                 movement_type: str = "simple",
                 world_and_level: Optional[Tuple[int, int]] = None,
                 idle_frames_threshold: Optional[int] = 1250,
                 colorful_rendering: bool = True,
    ) -> None:
        assert stack_mode in ("first_and_last", "all")
        self._stack_mode = stack_mode

        env_name = (f"SuperMarioBros" if world_and_level is None
                    else "SuperMarioBros-%d-%d" % world_and_level)
        env_name += f"-v{int(black_background)}"
        self._smb_env = gym_super_mario_bros.make(env_name)
        self._smb_env = JoypadSpace(self._smb_env,
                                    MOVEMENTS_TYPES[movement_type])

        self._actions_queue = []
        self._colorful_env = None
        if (grayscale or black_background) and colorful_rendering:
            self._colorful_env = gym_super_mario_bros.make(
                "SuperMarioBros-%d-%d-v0" % world_and_level
            )
            self._colorful_env = JoypadSpace(self._colorful_env,
                                             MOVEMENTS_TYPES[movement_type])

        self._stack_func = stack_func
        self._grayscale = grayscale

        self._score_weight = in_game_score_weight
        self._idle_frames_threshold = idle_frames_threshold

        self._last_score = 0
        self._last_x = 40
        self._idle_counter = 0

        self._rescale_pc = img_rescale_pc
        self._skip_frames = skip_frames

        self._obs_shape = self.reset().observation.shape
        self._num_actions = self._smb_env.action_space.n

    def reset(self):
        """ Returns the first `TimeStep` of a new episode. """
        self._smb_env.reset()
        self._last_score = 0
        self._last_x = 40
        self._idle_counter = 0

        self._actions_queue = []
        if self._colorful_env is not None:
            self._colorful_env.reset()

        return dm_env.restart(self.step(0).observation)

    def _is_idle(self, info):
        if self._idle_frames_threshold is None:
            return False

        x = info["x_pos"]
        delta_x = x - self._last_x
        self._last_x = x

        if abs(delta_x) < 1:
            self._idle_counter += 1
            return self._idle_counter > self._idle_frames_threshold

        self._idle_counter = 0
        return False

    def step(self, action) -> TimeStep:
        """ Updates the environment's state. """
        # NOTE:
        # The gym_super_mario_bros environment reuses the numpy array it
        # returns as observation. When stacking observations, this might be
        # a source of bugs (all observations in the stack might be representing
        # the same, final frame!), so always copy the arrays when doing that.
        # The observation arrays are already being copied inside
        # `self._preprocess_img`, so no explicit copying is needed here.

        action = int(action)
        initial_img, total_reward, done, info = self._smb_env.step(action)
        self._actions_queue.append(action)
        done = done or self._is_idle(info)

        # Skipping frames:
        if self._skip_frames > 0:
            imgs = [self._process_img(initial_img)]
            skip_count = 0
            while skip_count < self._skip_frames:
                skip_count += 1
                if not done:
                    last_img, reward, done, info = self._smb_env.step(action)
                    self._actions_queue.append(action)
                    done = done or self._is_idle(info)
                    total_reward += reward
                else:
                    last_img = np.zeros_like(initial_img)

                if self._stack_mode == "all" or skip_count == self._skip_frames:
                    imgs.append(self._process_img(last_img))

            obs = self._stack_func(imgs)
        # Single frame:
        else:
            obs = self._process_img(initial_img)

        score_diff = info["score"] - self._last_score
        self._last_score = info["score"]
        total_reward = np.float64(total_reward
                                  + self._score_weight * score_diff)

        if done:
            return dm_env.termination(reward=total_reward, observation=obs)
        return dm_env.transition(reward=total_reward, observation=obs)

    def observation_spec(self):
        return dm_env.specs.BoundedArray(shape=self._obs_shape,
                                         dtype=np.float32, name="image",
                                         minimum=0, maximum=1)

    def action_spec(self):
        return dm_env.specs.DiscreteArray(dtype=np.int32, name="action",
                                          num_values=self._num_actions)

    def _process_img(self, img):
        img = np.divide(img, 255)
        img = img[50:, :, :]

        if abs(self._rescale_pc - 1) > 1e-2:
            img = rescale(img, scale=self._rescale_pc, multichannel=True)

        if self._grayscale:
            img = img @ RGB2GRAY_COEFFICIENTS

        return img.astype(np.float32, copy=True)

    def render(self, mode="human", return_all_imgs=False):
        if return_all_imgs:
            assert self._colorful_env is not None and mode == "rgb_array", (
                "The option 'return_all_imgs' is valid only when using "
                "colorful rendering and rgb array mode!"
            )

        # Regular rendering:
        if self._colorful_env is None:
            return self._smb_env.render(mode)

        # Colorful rendering:
        img_list = []
        for action in self._actions_queue:
            self._colorful_env.step(action)
            if return_all_imgs:
                # NOTE: make sure a copy of the returned rgb array is made!
                img_list.append(self._colorful_env.render(mode).copy())

        self._actions_queue = []
        return img_list if return_all_imgs else self._colorful_env.render(mode)

    def plot_obs(self, obs):
        plt.imshow(obs, cmap="gray" if self._grayscale else None)
        plt.show()

    def close(self):
        self._smb_env.close()
