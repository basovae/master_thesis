from pga_map_elites import main, config

# Your environment
class DummyPortfolioEnv:
    def __init__(self):
        self.state_dim = 10
        self.action_dim = 5
        self.max_steps = 50
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.state_dim)
    
    def step(self, action):
        self.step_count += 1
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn() * 0.01
        done = self.step_count >= self.max_steps
        info = {'portfolio_return': reward}
        return next_state, reward, done, info

env = DummyPortfolioEnv()
config["state_dim"] = env.state_dim
config["action_dim"] = env.action_dim

archive = main(env, config)