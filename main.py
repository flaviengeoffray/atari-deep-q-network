import gymnasium as gym
import torch

from dqn.env import init_env, display_env
from dqn.agent import DQNAgent
from dqn.dqn import DQN
from dqn.train import TrainingConfig, training, evaluate


def load_checkpoint(model: DQN, path: str):
    model.load_state_dict(torch.load(path))
    model.eval()

training_config = TrainingConfig(
    nb_episodes=1000,
    nb_steps_per_episode=10000,
    memory_capacity=300000,
    batch_size=32,
    train_frequency=4,
    lr=0.00025,
    gamma=0.99,
    agent_epsilon_start=1.0,
    agent_epsilon_decay=1000000,
    agent_epsilon_end=0.01,
    target_update_frequency=10000,
    checkpoint_frequency=100
)

env = gym.make("PongNoFrameskip-v4")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, 4)
training(env, training_config)

# agent = DQNAgent(DQN(env.action_space.n), env.action_space.n, epsilon_start=0.0, epsilon_decay=1, epsilon_end=0.0)
# load_checkpoint(agent.model, "checkpoints/dqn_checkpoint_episode_1000.pth")
# avg_reward = evaluate(env, agent, n_episodes=10)
# print(f"Average Reward over 10 episodes: {avg_reward}")
