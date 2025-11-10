import ale_py # Import les environnements de la lib ALE dans gymnasium
import gymnasium as gym
import typing as t
import numpy as np
import matplotlib.pyplot as plt

from dataprep import preprocess_frame

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, 4)

n_actions = env.action_space.n
print(n_actions)

env.reset()

env.render()

env.close()


# To record a video
# env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
# env = gym.wrappers.RecordVideo(
#     env,
#     episode_trigger=lambda num: num % 2 == 0,
#     video_folder="saved-video-folder",
#     name_prefix="video-",
# )
# for episode in range(10):
#     obs, info = env.reset()
#     episode_over = False

#     while not episode_over:
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)

#         episode_over = terminated or truncated

# env.close()
