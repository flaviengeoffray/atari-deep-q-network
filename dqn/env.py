import gymnasium as gym
import numpy as np

import ale_py


def init_env(env_name: str, render_mode="rgb_array") -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
    
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Standard Atari Preprocessing (grayscale, resize 84x84, frame skip, normalisation)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,           
        frame_skip=4,          
        screen_size=84,        
        grayscale_obs=True,    
        scale_obs=True         # Normalize [0, 255] -> [0, 1]
    )
    
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    obs = np.array(obs, dtype=np.float32)
    # obs = obs.astype(np.float32) / 255.0  # Normalize pixel values
    return obs
