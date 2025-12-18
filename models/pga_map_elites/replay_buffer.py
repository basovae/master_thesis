import numpy as np
import torch


class ReplayBuffer:
    """
    FIFO Replay Buffer for experience storage
    Max size: 10^6 (paper default)
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    
    def add(self, transitions):
        l = len(transitions[0])
        idx = np.arange(self.ptr, self.ptr + l) % self.max_size
        self.state[idx] = transitions[0]
        self.action[idx] = transitions[1]
        self.next_state[idx] = transitions[2]
        self.reward[idx] = transitions[3]
        self.not_done[idx] = 1. - transitions[4]

        self.ptr = (self.ptr + l) % self.max_size
        self.size = min(self.size + l, self.max_size)
        self.additions += 1

    def add_batch(self, transitions):
        """Add batch of transitions from episode evaluation"""
        for t in transitions:
            self.add(*t)

    def sample(self, batch_size):
        """Uniform random sampling"""
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.done[ind])
        )
    