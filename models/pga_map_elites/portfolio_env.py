"""
Portfolio Environment for PGA-MAP-Elites

Based on QDgym interface (Nilsson & Cully, 2021) and FinRL patterns (Liu et al., 2020).
Behavior descriptors follow Gašperov et al. (2023): volatility and diversification.
"""

import numpy as np
import pandas as pd


class PortfolioEnv:
    """
    Simple portfolio environment compatible with PGA-MAP-Elites.
    
    State: flattened lookback window of returns
    Action: portfolio weights (softmax normalized)
    Reward: portfolio return - transaction costs
    """
    
    def __init__(self, data_path, lookback=20, episode_len=50, commission=0.0025):
        # Load data
        if isinstance(data_path, str):
            df = pd.read_csv(data_path, index_col=0)
            self.data = df.values
        else:
            self.data = data_path.values if hasattr(data_path, 'values') else data_path
        
        self.n_assets = self.data.shape[1]
        self.lookback = lookback
        self.episode_len = episode_len
        self.commission = commission
        
        # Dimensions for network init (QDgym interface)
        self.state_dim = lookback * self.n_assets
        self.action_dim = self.n_assets
        
        # QDgym requires these
        self._max_episode_steps = episode_len
        self.tot_reward = 0.0
        self.desc = np.zeros(2)  # [volatility, diversification]
        
        # Internal state
        self._idx = 0
        self._step = 0
        self._last_weights = np.ones(self.n_assets) / self.n_assets
        self._returns = []
    
    def reset(self):
        """Reset environment, return initial state."""
        # Random starting point
        max_start = len(self.data) - self.lookback - self.episode_len - 1
        self._start = np.random.randint(0, max(1, max_start))
        self._idx = self._start + self.lookback
        self._step = 0
        
        # Reset tracking
        self.tot_reward = 0.0
        self.desc = np.zeros(2)
        self._last_weights = np.ones(self.n_assets) / self.n_assets
        self._returns = []
        
        return self._get_state()
    
    def step(self, action):
        """Execute action, return (state, reward, done, info)."""
        # Softmax normalize to get valid weights
        action = action - np.max(action)  # numerical stability
        weights = np.exp(action) / np.sum(np.exp(action))
        
        # Portfolio return
        day_return = self.data[self._idx]
        portfolio_return = np.dot(weights, day_return)
        
        # Transaction cost
        turnover = np.sum(np.abs(weights - self._last_weights))
        cost = self.commission * turnover
        
        reward = portfolio_return - cost
        
        # Update state
        self._returns.append(portfolio_return)
        self._last_weights = weights.copy()
        self._idx += 1
        self._step += 1
        self.tot_reward += reward
        
        # Update behavior descriptor
        self._update_desc(weights)
        
        # Check done
        done = (self._step >= self.episode_len) or (self._idx >= len(self.data) - 1)
        
        info = {
            'portfolio_return': portfolio_return,
            'volatility': self.desc[0],
            'diversification': self.desc[1],
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """Get flattened lookback window as state."""
        start = self._idx - self.lookback
        return self.data[start:self._idx].flatten().astype(np.float32)
    
    def _update_desc(self, weights):
        """Update behavior descriptor: [volatility, diversification] normalized to [0,1]."""
        # Volatility = std of returns so far
        vol = np.std(self._returns) if len(self._returns) > 1 else 0.0
        div = 1.0 - np.sum(weights ** 2)
        
        # Normalize
        vol_norm = np.clip(vol / 0.03, 0.0, 1.0)  # Changed from 0.05
        div_norm = div / (1.0 - 1.0/self.n_assets)
        
        self.desc = np.array([vol_norm, div_norm])