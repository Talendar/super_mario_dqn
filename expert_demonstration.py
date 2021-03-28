"""
TODO
"""

import pygame
import time
import numpy as np
from skimage.transform import resize as img_resize

_ACTION_MAP = {
    "NOOP": 0,
    "right": 1,
    "right + A": 2,
    "right + B": 3,
    "right + A + B": 4,
    "A": 5,
    "left": 6,
}


def _freeze():
    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            time.sleep(0.2)
            break


def _get_action():
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:
        time.sleep(0.2)
        _freeze()

    if keys[pygame.K_RIGHT] and keys[pygame.K_UP] and keys[pygame.K_b]:
        return _ACTION_MAP["right + A + B"]
    if keys[pygame.K_RIGHT] and keys[pygame.K_UP]:
        return _ACTION_MAP["right + A"]
    if keys[pygame.K_RIGHT] and keys[pygame.K_b]:
        return _ACTION_MAP["right + B"]
    if keys[pygame.K_RIGHT]:
        return _ACTION_MAP["right"]
    # if keys[pygame.K_LEFT]:
    #     return _ACTION_MAP["left"]
    # if keys[pygame.K_UP]:
    #     return _ACTION_MAP["A"]

    return 0


def _render(display, screen_img):
    img2draw = img_resize(image=np.flipud(np.rot90(screen_img)),
                          output_shape=(500, 500),
                          preserve_range=True)

    main_surface = pygame.surfarray.make_surface(img2draw)
    display.fill(color="black")
    display.blit(main_surface, [0, 0])
    pygame.display.update()


def human_play(env, num_episodes):
    data = []
    display = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()

    for episode in range(num_episodes):
        timestep = env.reset()
        episode_data = {"first": timestep, "mid": []}
        episode_reward = 0.0

        # Rendering 1st frame:
        _render(display, screen_img=env.render(mode="rgb_array"))
        _freeze()

        while not timestep.last():
            # Acting:
            action = _get_action()
            timestep = env.step(action)

            episode_data["mid"].append((action, timestep))
            episode_reward += timestep.reward

            # Rendering:
            _render(display, screen_img=env.render(mode="rgb_array"))
            clock.tick(15)

        data.append(episode_data)
        print(f"Episode reward: {episode_reward}")

    return data
