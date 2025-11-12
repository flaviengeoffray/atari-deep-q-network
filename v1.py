# import torch
# import torch.nn as nn
# import gymnasium as gym
# import typing as t
# import numpy as np
# import tqdm
# import random
# from collections import namedtuple, deque

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'reward', 'next_state'))

# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)


# class DQN(nn.Module):
#     def __init__(self, n_actions: int):
#         super(DQN, self).__init__()
#         # "The input is an 84x84x4 image"
#         # "The first hidden layer convolves 16 filters of 8x8 with stride 4"
#         self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
#         # "Applies a rectifier nonlinearity"
#         self.relu1 = nn.ReLU()
#         # "The second hidden layer convolves 32 filters of 4x4 with stride 2 + ReLU"
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
#         self.relu2 = nn.ReLU()
#         # "The final hidden layer is fully-connected with 256 units + ReLU"
#         self.fc1 = nn.Linear(in_features=32 * 9 * 9, out_features=256)
#         self.relu3 = nn.ReLU()
#         self.output = nn.Linear(in_features=256, out_features=n_actions)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.output(x)
#         return x
    

# class DQNAgent:
#     def __init__(self, model: DQN, n_actions: int, epsilon_start: float, epsilon_decay: float, epsilon_end: float):
#         self.model = model
#         self.n_actions = n_actions
#         self.epsilon = epsilon_start
#         self.epsilon_start = epsilon_start
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_end = epsilon_end
#         self.actions = list(range(n_actions))
#         self.timestep = 0
#         self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#     def reset(self):
#         self.epsilon = self.epsilon_start
#         self.timestep = 0        

#     def get_best_action(self, state: np.ndarray):
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             q_values = self.model(state_tensor)
#         best_action = torch.argmax(q_values).item()
#         return best_action

#     def get_action(self, state: np.ndarray):
#         self.epsilon = max(
#             self.epsilon_end,
#             self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.timestep / self.epsilon_decay)
#         )
#         self.timestep += 1

#         if random.random() < self.epsilon:
#             return random.choice(self.actions)
#         else:
#             return self.get_best_action(state)
            
        


# def training(env: gym.Env):
    
#     nb_episodes = 1000
#     nb_steps_per_episode = 10000
#     memory_capacity = 100000

#     batch_size = 32
#     train_frequency = 4
#     lr=0.00025

#     total_rewards = []
#     total_losses = []

#     q_network = DQN(env.action_space.n)
#     target_network = DQN(env.action_space.n)
#     target_network.load_state_dict(q_network.state_dict())

#     agent = DQNAgent(q_network, env.action_space.n, epsilon_start=1.0, epsilon_decay=500000, epsilon_end=0.1)

#     memory = ReplayMemory(memory_capacity)

#     optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)

#     progress_bar = tqdm.tqdm(total=nb_episodes)
#     for episode in range(nb_episodes):

#         total_reward = 0.0
#         total_loss = 0.0

#         state, _ = env.reset()
        
#         for t in range(nb_steps_per_episode):

#             random_action = agent.get_action(state)

#             next_state, reward, terminated, truncated, _ = env.step(random_action)
#             reward = torch.tensor([reward], dtype=torch.float32).to(agent.device)

#             done = terminated or truncated
#             next_state = None if done else next_state

#             memory.push(state, random_action, reward, next_state)

#             state = next_state

#             total_reward += reward.item()

#             if len(memory) >= batch_size and t % train_frequency == 0:
#                 transitions = memory.sample(batch_size)
#                 batch = Transition(*zip(*transitions))

#                 non_final_mask = torch.tensor(
#                     tuple(map(lambda s: s is not None, batch.next_state)),
#                     device=agent.device,
#                     dtype=torch.bool,
#                 )
#                 non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

#                 state_batch = torch.stack(batch.state)
#                 action_batch = torch.tensor(batch.action, device=agent.device).unsqueeze(1)
#                 reward_batch = torch.cat(batch.reward)

#                 state_action_values = q_network(state_batch).gather(1, action_batch)

#                 next_state_values = torch.zeros(batch_size, device=agent.device)
#                 with torch.no_grad():
#                     next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0]
#                 expected_state_action_values = reward_batch + (0.99 * next_state_values)

#                 criterion = nn.SmoothL1Loss()
#                 loss = criterion(state_action_values.squeeze(), expected_state_action_values)
#                 total_loss += loss.item()

#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0) # Gradient clipping
#                 optimizer.step()
            
#             target_net_state_dict = target_network.state_dict()
#             q_net_state_dict = q_network.state_dict()
#             for key in q_net_state_dict:
#                 target_net_state_dict[key] = q_net_state_dict[key]*0.005 + target_net_state_dict[key]*(1-0.005)
#             target_network.load_state_dict(target_net_state_dict)

#             if done:
#                 break
        
#         total_losses.append(total_loss)
#         total_rewards.append(total_reward)
            
#         progress_bar.update(1)
#     progress_bar.close()
            


# def evaluate(env: gym.Env, agent: DQNAgent, n_episodes: int = 5) -> float:
#     total_rewards = []
#     for _ in range(n_episodes):
#         state, _ = env.reset()
#         total_reward = 0.0
#         while True:
#             action = agent.get_best_action(state)  # Pas d'exploration
#             state, reward, terminated, truncated, _ = env.step(action)
#             total_reward += reward
#             if terminated or truncated:
#                 break
#         total_rewards.append(total_reward)
#     return np.mean(total_rewards)
