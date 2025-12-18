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

    def add(self, state, action, next_state, reward, done):
        """Add single transition"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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
    

class ReplayBuffer:
    """
    Simple FIFO replay buffer.
    
    📝 PAPER CHECK: Uses 10^6 max size, FIFO replacement
    📝 YOUR CODE: Likely already has this
    
    🎓 SIMPLIFICATION: Using deque instead of numpy arrays
       (Simpler but slightly slower - fine for learning)
    """
    def __init__(self, max_size=100000):  # 🎓 Reduced from 1M for testing
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.FloatTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch]).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
