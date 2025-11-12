import torch

from dqn.env import init_env
from dqn.agent import DQNAgent
from dqn.dqn import DQN
from dqn.train import training, evaluate


def load_checkpoint(model: DQN, path: str):
    model.load_state_dict(torch.load(path))
    model.eval()


env = init_env("ALE/Pong-v5", render_mode="human")
training(env)

agent = DQNAgent(DQN(env.action_space.n), env.action_space.n, epsilon_start=0.0, epsilon_decay=1, epsilon_end=0.0)
load_checkpoint(agent.model, "checkpoints/dqn_checkpoint_episode_1000.pth")
avg_reward = evaluate(env, agent, n_episodes=10)
print(f"Average Reward over 10 episodes: {avg_reward}")
