import numpy as np
import pandas as pd
from .pga_map_elites import main, config

class PortfolioEnv:
    def __init__(self, data_path, lookback=20, episode_len=50):
        df = pd.read_csv(data_path, index_col=0)
        self.returns = df.values
        self.n_assets = df.shape[1]
        self.lookback = lookback
        self.episode_len = episode_len
        
        self.state_dim = lookback * self.n_assets
        self.action_dim = self.n_assets
        
    def reset(self):
        max_start = len(self.returns) - self.lookback - self.episode_len - 1
        self.start = np.random.randint(0, max_start)
        self.step_count = 0
        return self.returns[self.start:self.start + self.lookback].flatten()
    
    def step(self, action):
        weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        idx = self.start + self.lookback + self.step_count
        reward = np.dot(weights, self.returns[idx])
        
        self.step_count += 1
        done = self.step_count >= self.episode_len
        
        next_idx = self.start + self.step_count
        next_state = self.returns[next_idx:next_idx + self.lookback].flatten()
        
        return next_state, reward, done, {'portfolio_return': reward}


# Only run when executed directly, not when imported
if __name__ == "__main__":
    env = PortfolioEnv("/Users/ekaterinabasova/Desktop/untitled folder/master_thesis/data.csv")
    
    config["state_dim"] = env.state_dim
    config["action_dim"] = env.action_dim
    config["neurons_list"] = [256, 256]
    
    archive = main(env, config)