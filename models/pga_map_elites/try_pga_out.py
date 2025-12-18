import numpy as np

from pga_map_elites import PGAMAPElites


if __name__ == "__main__":
    """
    Example of how to use PGA-MAP-Elites.
    
    🎓 YOU NEED TO:
    1. Replace DummyEnv with your portfolio environment
    2. Customize behavior descriptor computation
    3. Adjust hyperparameters after initial testing
    """
    
    # Dummy environment for testing
    class DummyEnv:
        def __init__(self):
            self.state_dim = 10  # e.g., 10 features
            self.action_dim = 5  # e.g., 5 assets
            self.max_steps = 50
            self.step_count = 0
        
        def reset(self):
            self.step_count = 0
            return np.random.randn(self.state_dim)
        
        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(self.state_dim)
            reward = np.random.randn() * 0.01  # Random reward for testing
            done = self.step_count >= self.max_steps
            info = {
                'volatility': np.std(action),
                'diversification': 1 - np.sum(action**2)  # Inverse Herfindahl
            }
            return next_state, reward, done, info
    
    # Create environment and algorithm
    env = DummyEnv()
    
    pga_me = PGAMAPElites(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_niches=50,          # Start small for testing!
        behavior_dim=2,
        bd_bounds=((0, 0.5), (0, 1)),  # volatility, diversification bounds
        batch_size=10,
        random_init=50,
        n_crit=50,
        n_grad=5
    )
    
    # Run algorithm
    print("Starting PGA-MAP-Elites...")
    archive = pga_me.run(env, max_evals=500, verbose=True)
    
    print(f"\nFinal Results:")
    print(f"  Coverage: {archive.coverage:.1%}")
    print(f"  Max Fitness: {archive.max_fitness:.4f}")
    print(f"  Policies found: {sum(1 for p in archive.policies if p is not None)}")