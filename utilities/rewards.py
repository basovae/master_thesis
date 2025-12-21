"""
Reward functions for portfolio optimization.

Based on Gašperov et al. (2025) "Quality-diversity and Novelty Search 
for Portfolio Optimization and Beyond", Computational Economics.

Three target functions:
- Profit: Pure return maximization
- Sharpe Ratio: Risk-adjusted performance (Section 3, Eq. 7)
- Risk-Adjusted Profit: Tunable risk preference
"""

import torch


def profit(portfolio_returns: torch.Tensor) -> torch.Tensor:
    """Average profit (expected return).
    
    Args:
        portfolio_returns: Daily portfolio returns
        
    Returns:
        Mean return
    """
    return torch.mean(portfolio_returns)


def sharpe_ratio(
    portfolio_returns: torch.Tensor,
    risk_free_rate: float = 0.0,
) -> torch.Tensor:
    """Sharpe ratio: (Rp - Rf) / σp
    
    From paper Section 3: "Optimality is defined in terms of a 
    risk-adjusted return metric, such as the Sharpe ratio."
    
    Args:
        portfolio_returns: Daily portfolio returns, shape (batch, lookback) or (lookback,)
        risk_free_rate: Risk-free rate (default 0)
        
    Returns:
        Sharpe ratio per sample
    """
    avg_return = torch.mean(portfolio_returns, dim=-1)
    volatility = torch.std(portfolio_returns, dim=-1, correction=0)
    
    return (avg_return - risk_free_rate) / (volatility + 1e-8)


def risk_adjusted_profit(
    portfolio_returns: torch.Tensor,
    risk_preference: float = -0.5,
) -> torch.Tensor:
    """Risk-adjusted profit: μ + λσ
    
    Args:
        portfolio_returns: Daily portfolio returns
        risk_preference: Lambda coefficient (negative = risk-averse)
        
    Returns:
        Risk-adjusted profit
    """
    avg_return = torch.mean(portfolio_returns)
    volatility = torch.std(portfolio_returns, correction=0)
    
    return avg_return + risk_preference * volatility


# Simple dispatcher
REWARD_FUNCTIONS = {
    "profit": profit,
    "sharpe": sharpe_ratio,
    "risk_adjusted": risk_adjusted_profit,
}


def compute_reward(
    portfolio_returns: torch.Tensor,
    reward_type: str = "sharpe",
    **kwargs,
) -> torch.Tensor:
    """Compute reward using specified function.
    
    Args:
        portfolio_returns: Daily portfolio returns
        reward_type: One of "profit", "sharpe", "risk_adjusted"
        **kwargs: Additional arguments (risk_free_rate, risk_preference)
        
    Returns:
        Reward value
    """
    if reward_type not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                        f"Choose from {list(REWARD_FUNCTIONS.keys())}")
    
    func = REWARD_FUNCTIONS[reward_type]
    
    # Filter kwargs to only those the function accepts
    if reward_type == "profit":
        return func(portfolio_returns)
    elif reward_type == "sharpe":
        return func(portfolio_returns, kwargs.get("risk_free_rate", 0.0))
    else:  # risk_adjusted
        return func(portfolio_returns, kwargs.get("risk_preference", -0.5))