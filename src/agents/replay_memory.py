import random
from collections import deque


class ReplayMemory:
    def __init__(self, max_length):
        self.memory = deque([], maxlen=max_length)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
