
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass

import gymnasium as gym

from dqn.dqn import DQN
from dqn.memory import ReplayMemory, Transition


class DQNAgent:
    def __init__(self, n_actions: int,
                device: torch.device,
                memory_capacity: int = 100000,
                epsilon_start: float = 1.0,
                epsilon_decay: int = 500000,
                epsilon_end: float = 0.1,
                gamma: float = 0.99,
                lr: float = 0.00025,
                batch_size: int = 32,
                ):

        self.device = device

        self.policy_network = DQN(n_actions).to(device)
        self.target_network = DQN(n_actions).to(device)
        self.update_target_network()
        self.target_network.eval()

        self.memory = ReplayMemory(memory_capacity)

        self.n_actions = n_actions
        self.actions = list(range(n_actions))

        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.timestep = 0

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def reset(self):
        self.epsilon = self.epsilon_start
        self.timestep = 0        

    def get_best_action(self, state: np.ndarray):
        state_tensor = torch.FloatTensor(np.array([state])).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        best_action = torch.argmax(q_values).item()
        return best_action

    def get_action(self, state: np.ndarray):

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.timestep / self.epsilon_decay)
        )
        self.timestep += 1

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.get_best_action(state)
            