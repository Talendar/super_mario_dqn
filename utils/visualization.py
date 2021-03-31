import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame
import skvideo.io
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.filters import gaussian
from skimage.transform import resize


def _format_pygame_img(img_array):
    return np.flipud(np.rot90(img_array))


def make_saliency_map(input_img, gradients,
                      convert_to_uint8: bool = True,
                      resize_dims: Optional[Tuple] = None,
                      pygame_surface: bool = False):
    assert input_img.shape == gradients.shape
    gradients = tf.abs(gradients)
    # gradients = tf.nn.relu(gradients)
    gradients = tf.math.divide(tf.subtract(gradients,
                                           tf.reduce_min(gradients)),
                               tf.subtract(tf.reduce_max(gradients),
                                           tf.reduce_min(gradients)))

    t = tf.math.sqrt(tf.add(tf.math.reduce_std(gradients),
                            tf.reduce_mean(gradients)))
    gradients = tf.multiply(
        gradients,
        tf.cast(gradients > t, dtype=tf.float32)
    )
    gradients = gaussian(gradients, sigma=0.5 + t.numpy())

    saliency_map = tf.stack([gradients, input_img, input_img], axis=-1)

    if convert_to_uint8:
        saliency_map = tf.cast(tf.multiply(saliency_map, 255), dtype=tf.uint8)

    if resize_dims is not None:
        saliency_map = resize(saliency_map,
                              output_shape=[resize_dims[1], resize_dims[0], 3],
                              preserve_range=True)
    if pygame_surface:
        saliency_map = pygame.surfarray.make_surface(
            _format_pygame_img(saliency_map))

    return saliency_map


def _make_q_values_plot(q_values,
                        actions_names,
                        plt_objs,
                        resize_dims=None,
                        pygame_surface=False):
    fig, axis, canvas = tuple(map(plt_objs.get, ("fig", "axis", "canvas")))

    axis.clear()
    bar_list = axis.bar(x=actions_names, height=q_values)
    bar_list[np.argmax(q_values)].set_color("orange")

    canvas.draw()
    fig_width, fig_height = fig.get_size_inches() * fig.get_dpi()

    q_values_plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    q_values_plot = q_values_plot.reshape(int(fig_height), int(fig_width), 3)

    if resize_dims is not None:
        q_values_plot = resize(q_values_plot,
                               output_shape=[resize_dims[1], resize_dims[0], 3],
                               preserve_range=True)
    if pygame_surface:
        q_values_plot = pygame.surfarray.make_surface(
            _format_pygame_img(q_values_plot))

    return q_values_plot


def visualize_policy(policy,
                     env,
                     num_episodes: int,
                     fps: int = 0,
                     epsilon_greedy: float = 0.025,
                     plot_extras: bool = False,
                     save_video: bool = False,
                     save_video_to: Optional[str] = None):
    if plot_extras or save_video:
        plt_objs = {"fig": plt.figure()}
        plt_objs["axis"] = plt_objs["fig"].add_subplot(1, 1, 1)
        plt_objs["canvas"] = FigureCanvas(plt_objs["fig"])

        display = pygame.display.set_mode((960, 540))
        clock = pygame.time.Clock()

        if save_video:
            img_cache = []

    env.reset()
    for episode in range(num_episodes):
        obs = env.reset().observation
        episode_reward = 0.0

        done = False
        while not done:
            # Traditional rendering:
            if not plot_extras and not save_video:
                env.render(mode="human")
                time.sleep(1 / fps)
            # Getting environment's image(s):
            else:
                env_rgb = env.render(mode="rgb_array",
                                     return_all_imgs=save_video)
                if type(env_rgb) != list:
                    env_rgb = [env_rgb]

            # Calculating Q-values:
            obs = tf.Variable(tf.expand_dims(obs, axis=0))
            with tf.GradientTape() as tape:
                tape.watch(obs)
                q_values = policy(obs)[0]
                max_q = q_values[tf.argmax(q_values)]

            # Plotting extras / preparing video:
            if plot_extras or save_video:
                # Q-values and saliency map surfaces:
                q_plot_surface = _make_q_values_plot(
                    q_values=q_values,
                    actions_names=["NOOP", "=>", "=> /\\", "===>", "===> /\\"],
                    plt_objs=plt_objs,
                    resize_dims=[320, 270], pygame_surface=True
                )
                saliency_map_surface = make_saliency_map(
                    input_img=obs[0, :, :, -1],
                    gradients=tape.gradient(max_q, obs)[0, :, :, -1],
                    resize_dims=[320, 270], pygame_surface=True
                )

                # Drawing:
                for img in env_rgb:
                    env_surface = pygame.surfarray.make_surface(
                        resize(_format_pygame_img(img),
                               output_shape=[640, 540, 3],
                               preserve_range=True)
                    )

                    display.fill("black")
                    display.blit(source=env_surface, dest=(0, 0))
                    display.blit(source=q_plot_surface, dest=(640, 0))
                    display.blit(source=saliency_map_surface,
                                 dest=(640, 270))

                    if save_video:
                        rgb_array = pygame.surfarray.array3d(
                                        pygame.display.get_surface())
                        img_cache.append(np.flipud(np.rot90(rgb_array)))

                # Displaying:
                pygame.display.update()
                clock.tick(fps)

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

    # Generating video:
    if save_video:
        if save_video_to is None:
            Path("./videos").mkdir(exist_ok=True)
            date_and_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
            save_video_to = f"./videos/video_{date_and_time}.avi"

        skvideo.io.vwrite(fname=save_video_to,
                          videodata=img_cache,
                          outputdict={"-r": "30",
                                      "-vcodec": "libx264",
                                      "-b": "800k"})
