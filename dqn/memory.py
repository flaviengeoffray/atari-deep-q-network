import random
from collections import namedtuple, deque

random.seed(42)

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return Transition(*zip(*sample))

    def __len__(self):
        return len(self.memory)
