"""
Configuration file for experiment hyperparameters.

This file centralizes all hyperparameters for easy experimentation
and comparison across different models.
"""

# Data settings
DATA_CONFIG = {
    "lookback_window": 30,
    "forecast_window": 0,
    "batch_size": 1,
}

# Training settings
TRAINING_CONFIG = {
    "num_epochs": 50,
    "early_stopping": True,
    "patience": 2,
    "min_delta": 0,
    "seed": 42,
}

# DDPG-specific settings
DDPG_CONFIG = {
    "actor_lr": 0.05,
    "critic_lr": 0.01,
    "gamma": 1.0,
    "tau": 0.005,
    "soft_update": False,
    "l1_lambda": 0,
    "l2_lambda": 0,
    "weight_decay": 0,
    "risk_preference": -0.5,
}

# Deep Q-Learning settings
DQN_CONFIG = {
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "gamma": 0.99,
    "tau": 0.005,
    "soft_update": True,
    "l1_lambda": 0,
    "l2_lambda": 0,
    "weight_decay": 0,
    "risk_preference": -0.5,
    "num_action_samples": 10,
}

# Neural network architecture
NETWORK_CONFIG = {
    "hidden_sizes": [64, 32],
}

# Portfolio settings
PORTFOLIO_CONFIG = {
    "short_selling": False,
    "reduce_negatives": False,
}

# Evaluation settings
EVAL_CONFIG = {
    "verbose": 1,
}
