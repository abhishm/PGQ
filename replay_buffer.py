from collections import deque
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_items = 0
        self.buffer = deque()

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def add(self, item):
        if self.num_items < self.buffer_size:
            self.buffer.append(item)
            self.num_items += 1
        else:
            self.buffer.popleft()
            self.buffer.append(item)

    def add_items(self, items):
        for item in items:
            self.add(item)

    def add_batch(self, batch):
        keys = ["states", "actions", "rewards", "next_states", "dones"]
        items = []
        for i in range(len(batch["states"])):
            item = []
            for key in keys:
                item.append(batch[key][i])
            items.append(item)
        self.add_items(items)

    def sample_batch(self, batch_size):
        keys = ["states", "actions", "rewards", "next_states", "dones"]
        samples = self.sample(batch_size)
        samples = zip(*samples)
        batch = {key: np.array(value) for key, value in zip(keys, samples)}
        return batch

    def count(self):
        return self.num_items

    def erase(self):
        self.buffer = deque()
        self.num_items = 0
