import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Twin Q-networks in single class (TD3 official implementation)
    Architecture: [state_dim + action_dim] -> [256, 256, 1]
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture (identical structure, independent weights)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        """Returns both Q-values"""
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Returns only Q1 (used for actor updates)"""
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class Actor(nn.Module):
    """
    Deterministic policy network
    Architecture: [state_dim] -> [128, 128] -> [action_dim]

    Note: PGA-ME uses [128, 128], TD3 uses [256, 256]
    Paper recommends [128, 128] for controllers
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=(128, 128)):
        super(Actor, self).__init__()

        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.network(state)

    def get_params(self):
        """Flatten parameters for GA variation"""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_params(self, flat_params):
        """Unflatten parameters after GA variation"""
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view(p.shape))
            idx += numel


class PortfolioActor(nn.Module):
    """
    Portfolio policy network with constraints.
    Output: Portfolio weights that sum to 1 (long-only) or
            handle short positions with leverage constraints.
    """
    def __init__(self, state_dim, n_assets, hidden_sizes=(128, 128)):
        super().__init__()

        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, n_assets))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        raw_output = self.network(state)
        # Softmax ensures weights sum to 1 (long-only constraint)
        weights = F.softmax(raw_output, dim=-1)
        return weights