import gymnasium as gym
import torch
import numpy as np

import ale_py

def init_env(env_name: str) -> gym.Env:
    # env = gym.make(env_name, render_mode="rgb_array", repeat_action_probability=0.0) 
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayscaleObservation(env)
    # env = gym.wrappers.FrameStackObservation(env, 4)
    env = gym.make(env_name, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
    
    # Enregistre les stats d'épisode (reward, length)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Preprocessing Atari standard (grayscale, resize 84x84, frame skip, normalisation)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,           # Max random no-ops au début de chaque épisode
        frame_skip=4,          # Répète chaque action 4 fois
        screen_size=84,        # Resize à 84x84
        grayscale_obs=True,    # Convertit en grayscale
        scale_obs=True         # Normalise [0, 255] -> [0, 1] (float32)
    )
    
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Preprocess the observation for the DQN agent."""
    obs = np.array(obs, dtype=np.float32)  # Ensure type is float32
    # obs = obs.astype(np.float32) / 255.0  # Normalize pixel values
    return obs
