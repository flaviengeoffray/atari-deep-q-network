import torch

from dqn.env import init_env
from dqn.agent import DQNAgent
from dqn.dqn import DQN
from dqn.train import TrainingConfig, train, evaluate


def load_checkpoint(model: DQN, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

training_config = TrainingConfig(
    nb_episodes=5000,
    nb_steps_per_episode=10000,
    memory_capacity=100000,
    batch_size=32,
    train_frequency=4,
    lr=0.00025,
    gamma=0.99,
    agent_epsilon_start=1.0,
    agent_epsilon_decay=1000000,
    agent_epsilon_end=0.1,
    target_update_frequency=10000,
    checkpoint_frequency=100,
    learning_starts=10000,
)

# training_env = init_env("ALE/Pong-v5")
# for param in training_config.__dataclass_fields__.values():
#     print(f"{param.name}: {getattr(training_config, param.name)}")
# train(training_env, training_config)


env = init_env("ALE/Pong-v5")
agent = DQNAgent(
    DQN(env.action_space.n),
    env.action_space.n,
    epsilon_start=0.0,
    epsilon_decay=1,
    epsilon_end=0.0,
)
load_checkpoint(agent.model, "checkpoints/dqn_pong_5000.pth")
avg_reward = evaluate(env, agent, n_episodes=10, create_video=True)
print(f"Average Reward over 10 episodes: {avg_reward}")
