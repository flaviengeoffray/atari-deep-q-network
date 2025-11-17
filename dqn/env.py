import gymnasium as gym
import torch
import numpy as np

import ale_py

def init_env(env_name: str, render_mode: str = "rgb_array", repeat_action_probability: float = 0.25) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode, repeat_action_probability=repeat_action_probability) # No sticky actions
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def display_env(env: gym.Env):
    from gymnasium.utils.play import play
    play(env)
