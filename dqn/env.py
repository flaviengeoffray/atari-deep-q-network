import gymnasium as gym
import torch
import numpy as np

import ale_py

def init_env(env_name: str) -> gym.Env:
    env = gym.make(env_name, render_mode="rgb_array", repeat_action_probability=0.0) 
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def display_env(env: gym.Env):
    from gymnasium.utils.play import play
    play(env)

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Preprocess the observation for the DQN agent."""
    obs = np.array(obs)
    obs = obs.astype(np.float32) / 255.0  # Normalize pixel values
    return obs
