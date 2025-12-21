"""
PortfolioEnv: Hybrid Portfolio Optimization Environment

Combines:
- FinRL's PortfolioOptimizationEnv design patterns (Liu et al., 2020)
- QDgym interface for MAP-Elites compatibility (Nilsson & Cully, 2021)

Every design choice is traced to its source via [FINRL] or [QDGYM] tags.

References:
    [1] Liu et al. "FinRL: A Deep Reinforcement Learning Library for 
        Automated Stock Trading" (NeurIPS 2020 Workshop)
        https://github.com/AI4Finance-Foundation/FinRL
    
    [2] Nilsson & Cully. "Policy Gradient Assisted MAP-Elites" (GECCO 2021)
        https://github.com/ollenilsson19/QDgym
    
    [3] Gašperov et al. "Quality-Diversity Portfolio Optimization" (GECCO 2023)
        - Behavior descriptors: volatility and diversification
"""

import numpy as np
import pandas as pd


# =============================================================================
# [QDGYM] Space classes matching QDgym/Gym interface
# Source: QDgym uses gym.spaces, we replicate the minimal interface
# =============================================================================

class Box:
    """Minimal Box space matching gym.spaces.Box interface."""
    # [QDGYM] PGA-MAP-Elites accesses: space.shape[0], space.high[0]
    
    def __init__(self, low, high, shape):
        self.low = np.full(shape, low)
        self.high = np.full(shape, high)
        self.shape = shape


# =============================================================================
# [FINRL] + [QDGYM] Hybrid Environment
# =============================================================================

class PortfolioEnv:
    """
    Portfolio optimization environment for QD algorithms.
    
    State: Flattened lookback window of returns [FINRL pattern]
    Action: Portfolio weights, softmax-normalized [FINRL]
    Reward: Daily portfolio return minus transaction costs [FINRL]
    Descriptor: [volatility, diversification] for archive placement [QDGYM]
    
    Args:
        df: DataFrame with daily returns (rows=days, cols=assets)
        lookback: Number of historical days in state [FINRL: time_window]
        episode_len: Max steps per episode [QDGYM: _max_episode_steps]
        commission_fee_pct: Transaction cost as fraction [FINRL: 0.0025 default]
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        episode_len: int = 50,
        commission_fee_pct: float = 0.0025,  # [FINRL] Default 0.25%
    ):
        # ---------------------------------------------------------------------
        # [FINRL] Data handling - follows PortfolioOptimizationEnv pattern
        # ---------------------------------------------------------------------
        self.data = df.values if isinstance(df, pd.DataFrame) else df
        self.n_assets = self.data.shape[1]          # [FINRL] portfolio_size
        self.lookback = lookback                     # [FINRL] time_window
        self.commission_fee_pct = commission_fee_pct # [FINRL] comission_fee_pct
        
        # ---------------------------------------------------------------------
        # [QDGYM] Required attributes for PGA-MAP-Elites evaluation
        # Source: QDgym/envs/*.py and vectorized_env.py
        # ---------------------------------------------------------------------
        self._max_episode_steps = episode_len  # [QDGYM] Used for done_bool calc
        self.tot_reward = 0.0                  # [QDGYM] Cumulative fitness
        self.desc = np.zeros(2)                # [QDGYM] Behavior descriptor
        self.alive = True                      # [QDGYM] Survival flag
        self.T = 0                             # [QDGYM] Timestep counter
        
        # ---------------------------------------------------------------------
        # [QDGYM] Gym-compatible spaces for network initialization
        # Source: main.py lines 89-92 access these for Actor/Critic dims
        # ---------------------------------------------------------------------
        state_dim = lookback * self.n_assets   # [FINRL] Flattened observation
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(state_dim,)
        )
        self.action_space = Box(
            low=0.0,   # [FINRL] No short selling (low=0)
            high=1.0,  # [FINRL] Max weight = 1
            shape=(self.n_assets,)
        )
        
        # ---------------------------------------------------------------------
        # [FINRL] Portfolio state tracking
        # Source: PortfolioOptimizationEnv._final_weights pattern
        # ---------------------------------------------------------------------
        self._last_weights = np.ones(self.n_assets) / self.n_assets  # [FINRL]
        
        # ---------------------------------------------------------------------
        # [QDGYM] + [GASPEROV] Tracking for behavior descriptor computation
        # Descriptors: volatility (risk), diversification (concentration)
        # ---------------------------------------------------------------------
        self._returns_history = []
        self._weights_history = []
        
        # Episode position
        self._start_idx = 0
        self._current_idx = 0
    
    # =========================================================================
    # [FINRL] Softmax normalization - exact pattern from FinRL
    # Source: env_portfolio_optimization.py _softmax_normalization()
    # =========================================================================
    
    def _softmax_normalization(self, actions: np.ndarray) -> np.ndarray:
        """
        [FINRL] Normalize actions to valid portfolio weights via softmax.
        Ensures weights are positive and sum to 1.
        """
        # [FINRL] Numerical stability: subtract max before exp
        actions = actions - np.max(actions)
        numerator = np.exp(actions)
        denominator = np.sum(numerator)
        return numerator / denominator
    
    # =========================================================================
    # [FINRL] Transaction cost calculation
    # Source: env_portfolio_optimization.py, simplified from TRF model
    # =========================================================================
    
    def _calculate_transaction_cost(
        self, 
        old_weights: np.ndarray, 
        new_weights: np.ndarray
    ) -> float:
        """
        [FINRL] Calculate transaction cost from weight changes.
        Cost = commission_fee_pct * sum(|weight_changes|)
        """
        # [FINRL] Turnover = sum of absolute weight changes
        turnover = np.sum(np.abs(new_weights - old_weights))
        # [FINRL] Cost proportional to turnover
        cost = self.commission_fee_pct * turnover
        return cost
    
    # =========================================================================
    # [QDGYM] Required interface: reset()
    # Source: QDgym envs return state only (not tuple like Gym 0.26+)
    # =========================================================================
    
    def reset(self) -> np.ndarray:
        """
        [QDGYM] Reset environment to initial state.
        Returns: Initial observation (flattened lookback window)
        """
        # [QDGYM] Reset QD-specific tracking
        self.tot_reward = 0.0
        self.T = 0
        self.alive = True
        self.desc = np.zeros(2)
        
        # [FINRL] Reset portfolio to equal weights
        self._last_weights = np.ones(self.n_assets) / self.n_assets
        
        # [QDGYM] Reset history for descriptor computation
        self._returns_history = []
        self._weights_history = []
        
        # [FINRL] Random start position (within valid range)
        max_start = len(self.data) - self.lookback - self._max_episode_steps
        self._start_idx = np.random.randint(0, max(1, max_start))
        self._current_idx = self._start_idx + self.lookback
        
        # [FINRL] Return flattened observation
        return self._get_observation()
    
    # =========================================================================
    # [QDGYM] Required interface: step()
    # Source: QDgym step returns (state, reward, done, info)
    # =========================================================================
    
    def step(self, action: np.ndarray) -> tuple:
        """
        [QDGYM] Execute one step in environment.
        
        Args:
            action: Raw network output [QDGYM]
            
        Returns:
            tuple: (next_state, reward, done, info) [QDGYM format]
        """
        # ---------------------------------------------------------------------
        # [FINRL] Process action through softmax normalization
        # ---------------------------------------------------------------------
        weights = self._softmax_normalization(action)
        
        # ---------------------------------------------------------------------
        # [FINRL] Calculate portfolio return
        # Source: PortfolioOptimizationEnv step() reward calculation
        # ---------------------------------------------------------------------
        day_returns = self.data[self._current_idx]  # [FINRL] Current day returns
        portfolio_return = np.dot(weights, day_returns)  # [FINRL] Weighted sum
        
        # [FINRL] Subtract transaction costs
        transaction_cost = self._calculate_transaction_cost(
            self._last_weights, weights
        )
        reward = portfolio_return - transaction_cost
        
        # ---------------------------------------------------------------------
        # [QDGYM] Update QD-specific state
        # Source: QDgym environments track these for evaluation
        # ---------------------------------------------------------------------
        self.tot_reward += reward  # [QDGYM] Cumulative fitness
        self.T += 1                # [QDGYM] Timestep counter
        
        # ---------------------------------------------------------------------
        # [QDGYM] + [GASPEROV] Update behavior descriptor tracking
        # ---------------------------------------------------------------------
        self._returns_history.append(portfolio_return)
        self._weights_history.append(weights.copy())
        self._update_descriptor(weights)
        
        # ---------------------------------------------------------------------
        # [FINRL] Update portfolio state
        # ---------------------------------------------------------------------
        self._last_weights = weights.copy()
        self._current_idx += 1
        
        # ---------------------------------------------------------------------
        # [QDGYM] Check termination
        # Source: QDgym uses T >= _max_episode_steps or data exhaustion
        # ---------------------------------------------------------------------
        done = (
            self.T >= self._max_episode_steps or 
            self._current_idx >= len(self.data) - 1
        )
        
        # [QDGYM] Info dict with episode data
        info = {
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'weights': weights.copy(),
            'volatility': self.desc[0],
            'diversification': self.desc[1],
        }
        
        return self._get_observation(), reward, done, info
    
    # =========================================================================
    # [FINRL] Observation construction
    # Source: PortfolioOptimizationEnv observation patterns
    # =========================================================================
    
    def _get_observation(self) -> np.ndarray:
        """
        [FINRL] Construct observation from lookback window.
        Returns flattened array of shape (lookback * n_assets,)
        """
        start = self._current_idx - self.lookback
        end = self._current_idx
        # [FINRL] Flatten time window into 1D state vector
        obs = self.data[start:end].flatten()
        return obs.astype(np.float32)
    
    # =========================================================================
    # [QDGYM] + [GASPEROV] Behavior descriptor computation
    # Volatility: Gašperov et al. use std of returns
    # Diversification: 1 - Herfindahl index (concentration measure)
    # =========================================================================
    
    def _update_descriptor(self, weights: np.ndarray):
        """
        [QDGYM] Update behavior descriptor for archive placement.
        
        Descriptor dimensions following Gašperov et al. (2023):
        - dim 0: Portfolio volatility (std of returns)
        - dim 1: Diversification (1 - Herfindahl index)
        """
        # [GASPEROV] Volatility = standard deviation of realized returns
        if len(self._returns_history) > 1:
            volatility = np.std(self._returns_history)
        else:
            volatility = 0.0
        
        # [GASPEROV] Diversification = 1 - sum(weights^2)
        # Herfindahl index measures concentration; 1-H measures spread
        herfindahl = np.sum(weights ** 2)
        diversification = 1.0 - herfindahl
        
        # [QDGYM] Store in desc attribute for archive access
        self.desc = np.array([volatility, diversification])
    
    # =========================================================================
    # [QDGYM] Properties matching QDgym interface
    # =========================================================================
    
    @property
    def state_dim(self) -> int:
        """[QDGYM] State dimension for network initialization."""
        return self.observation_space.shape[0]
    
    @property
    def action_dim(self) -> int:
        """[QDGYM] Action dimension for network initialization."""
        return self.action_space.shape[0]


# =============================================================================
# Test / Usage Example
# =============================================================================

if __name__ == "__main__":
    # Create dummy data (100 days, 5 assets)
    np.random.seed(42)
    returns = np.random.randn(100, 5) * 0.02  # ~2% daily vol
    df = pd.DataFrame(returns, columns=[f'Asset_{i}' for i in range(5)])
    
    # Create environment
    env = PortfolioEnv(df, lookback=10, episode_len=20)
    
    print("=== Environment Properties ===")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Max episode steps: {env._max_episode_steps}")
    
    # Run episode
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    
    total_reward = 0
    for t in range(25):  # More than episode_len to test termination
        action = np.random.randn(env.action_dim)  # Random action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"\n=== Episode Complete at T={env.T} ===")
            break
    
    print(f"tot_reward (cumulative): {env.tot_reward:.6f}")
    print(f"Total reward (manual sum): {total_reward:.6f}")
    print(f"Match: {np.isclose(env.tot_reward, total_reward)}")
    print(f"\nFinal descriptor:")
    print(f"  Volatility: {env.desc[0]:.6f}")
    print(f"  Diversification: {env.desc[1]:.6f}")
    print(f"\nFinal weights: {info['weights']}")
    print(f"Weights sum: {info['weights'].sum():.6f}")