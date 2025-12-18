import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Portfolio policy network.
    
    📝 PAPER CHECK: Paper uses [128, 128] hidden layers
    📝 YOUR CODE: Likely similar, verify your hidden_dim
    
    🎓 SIMPLIFICATION: Using Sequential instead of separate layers
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Portfolio weights sum to 1
        )
    
    def forward(self, state):
        return self.net(state)
    
    # 🎓 ADDED FOR PGA-MAP-ELITES: Need to get/set params for GA variation
    def get_flat_params(self):
        """Flatten all parameters into single vector"""
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_flat_params(self, flat_params):
        """Set parameters from flat vector"""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(flat_params[idx:idx+size].view(p.shape))
            idx += size


class Critic(nn.Module):
    """
    Q-value network.
    
    📝 PAPER CHECK (TD3): Uses [256, 256] for critic
    📝 OFFICIAL TD3 CODE: Confirms [256, 256]
    
    🎓 SIMPLIFICATION: Your code has single critic, PGA-ME needs TWO critics
       But we can keep it simple by having one class return both Q1 and Q2
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network (🎓 THIS IS NEW - needed for TD3's clipped double Q)
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """Returns both Q values"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def Q1(self, state, action):
        """Returns only Q1 - used for actor updates"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)




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