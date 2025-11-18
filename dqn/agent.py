
import random
import numpy as np
import torch

from dqn.dqn import DQN

random.seed(42)

class DQNAgent:
    def __init__(self, model: DQN, n_actions: int, epsilon_start: float, epsilon_decay: float, epsilon_end: float, device: torch.device):
        self.model = model
        self.n_actions = n_actions
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.actions = list(range(n_actions))
        self.timestep = 0
        self.device = device

    def reset(self):
        self.epsilon = self.epsilon_start
        self.timestep = 0        

    def get_best_action(self, obs: np.ndarray):
        # obs is shape (84, 84, 4) and is already normalized
        state_tensor = torch.FloatTensor(np.array([obs])).to(self.device)
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
            