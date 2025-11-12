import gymnasium as gym

def init_env(env_name: str, render_mode: str = "rgb_array") -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env
