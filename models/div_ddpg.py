"""
DDPG for Portfolio Optimization with Optional DDES

This file implements:
- Standard DDPG (Lillicrap et al., 2015)
- DDPG with Diversity-Driven Exploration Strategy (Hong et al., 2018)

Toggle DDES via `use_ddes=True` in training config.

Key difference:
- DDPG: Uses Gaussian noise for exploration
- DDES-DDPG: Replaces noise with diversity term in actor loss
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Type

from copy import deepcopy

from utilities.data_processing import RLDataLoader
from utilities.metrics import RLEvaluator
from utilities.training import EarlyStopper, ReplayBuffer
from utilities.rewards import compute_reward


class DDPG:
    """DDPG for portfolio optimization with optional DDES.
    
    Args:
        lookback_window (int): Size of the lookback window for input data.
        predictor (Type[nn.Module]): The predictor class to use for the model.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        short_selling (bool, optional): Whether to allow short selling.
            Defaults to False.
        forecast_window (int, optional): Size of the forecast window.
            Defaults to 0.
        reduce_negatives (bool, optional): Whether to clamp negative weights.
            Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 42.
        **kwargs: Keyword arguments passed to the predictor.
    """
    def __init__(
        self,
        lookback_window: int,
        predictor: Type[nn.Module],
        batch_size: int = 1,
        short_selling: bool = False,
        forecast_window: int = 0,
        reduce_negatives: bool = False,
        verbose: int = 1,
        seed: int = 42,
        **kwargs,
    ):
        self.lookback_window = lookback_window
        self.batch_size = batch_size
        self.predictor = predictor
        self.predictor_kwargs = kwargs
        self.short_selling = short_selling
        self.forecast_window = forecast_window
        self.reduce_negatives = reduce_negatives
        self.verbose = verbose
        self.seed = seed

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
        optimizer: torch.optim = torch.optim.Adam,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        soft_update: bool = True,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        weight_decay: float = 0,
        gamma: float = 0.99,
        num_epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 5,
        min_delta: float = 0,
        noise: float = 0.2,
        # === DDES Parameters ===
        use_ddes: bool = False,
        ddes_alpha: float = 1.0,
        ddes_alpha_final: float = 0.0,
        ddes_scaling_method: str = "fixed",
        ddes_n_prior_samples: int = 16,
    ):
        """Trains the DDPG model.

        Args:
            train_data (pd.DataFrame): Training dataset.
            val_data (pd.DataFrame): Validation dataset.
            actor_lr (float): Learning rate for the actor.
            critic_lr (float): Learning rate for the critic.
            optimizer (torch.optim): Optimizer class.
            l1_lambda (float): L1 regularization for actor.
            l2_lambda (float): L2 regularization for actor.
            soft_update (bool): Whether to use soft updates for targets.
            tau (float): Soft update factor.
            risk_preference (float): Risk preference for reward (unused with Sharpe).
            weight_decay (float): Weight decay for critic optimizer.
            gamma (float): Discount factor.
            num_epochs (int): Number of training epochs.
            early_stopping (bool): Whether to use early stopping.
            patience (int): Early stopping patience.
            min_delta (float): Minimum change for early stopping.
            noise (float): Exploration noise std (ignored if use_ddes=True).
            use_ddes (bool): Enable Diversity-Driven Exploration Strategy.
            ddes_alpha (float): Diversity term weight.
            ddes_alpha_final (float): Final alpha for linear decay.
            ddes_scaling_method (str): One of "fixed", "linear_decay", "distance_based".
            ddes_n_prior_samples (int): Number of prior actions to sample.

        Returns:
            DDPG: The trained model instance.
        """
        self.val_data = val_data
        dataloader = RLDataLoader(train_data, val_data, shuffle=False)

        # Set data-related hyperparameters
        self.number_of_assets = dataloader.number_of_assets
        number_of_datapoints = self.lookback_window + self.forecast_window
        self.input_size = self.number_of_assets * number_of_datapoints
        self.output_size = self.number_of_assets

        # Build dataloaders
        train_loader, val_loader = dataloader(
            batch_size=self.batch_size,
            window_size=self.lookback_window,
            forecast_size=self.forecast_window,
        )

        # Initialize models
        if self.short_selling:
            activation = lambda x: x / torch.sum(x, dim=-1, keepdim=True)
        else:
            activation = nn.Softmax(dim=-1)
            
        self.actor = self.predictor(
            input_size=self.input_size,
            output_size=self.output_size,
            output_activation=activation,
            **self.predictor_kwargs,
            seed=self.seed,
        )
        critic = self.predictor(
            input_size=self.input_size + self.output_size,
            output_size=1,
            **self.predictor_kwargs,
            seed=self.seed,
        )

        # Run training loop
        trainer = DDPGTrainer(
            number_of_assets=self.number_of_assets,
            actor=self.actor,
            critic=critic,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            optimizer=optimizer,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            weight_decay=weight_decay,
            soft_update=soft_update,
            tau=tau,
            risk_preference=risk_preference,
            gamma=gamma,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            # DDES params
            use_ddes=use_ddes,
            ddes_alpha=ddes_alpha,
            ddes_alpha_final=ddes_alpha_final,
            ddes_scaling_method=ddes_scaling_method,
            ddes_n_prior_samples=ddes_n_prior_samples,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=self.verbose,
            num_epochs=num_epochs,
            noise=noise,
        )

        return self

    def evaluate(self, test_data: pd.DataFrame, dpo: bool = True) -> tuple:
        """Evaluates the DDPG model.

        Args:
            test_data (pd.DataFrame): Test dataset.
            dpo (bool, optional): Whether to evaluate DPO strategy. Defaults to True.

        Returns:
            tuple: ((SPO profit, SPO sharpe), (DPO profit, DPO sharpe)) if dpo=True
        """
        evaluator = RLEvaluator(
            actor=self.actor,
            train_data=self.val_data,
            test_data=test_data,
            forecast_size=self.forecast_window,
            reduce_negatives=self.reduce_negatives,
        )
        spo_results = evaluator.evaluate_spo(verbose=self.verbose)
        if dpo:
            dpo_results = evaluator.evaluate_dpo(
                interval=self.lookback_window,
                verbose=self.verbose
            )
            return spo_results, dpo_results
        else:
            return spo_results


class DDPGTrainer:
    """Trainer for DDPG with optional DDES.
    
    Standard DDPG uses Gaussian noise for exploration.
    With DDES enabled, exploration noise is replaced by a diversity term
    in the actor loss that encourages actions different from prior actions
    in the replay buffer (Hong et al., 2018).
    
    Actor loss:
        - DDPG:      L = -Q(s, π(s))
        - DDES-DDPG: L = -Q(s, π(s)) + α * diversity_loss
    
    Where diversity_loss = -mean(MSE(current_action, prior_actions))
    (negative because we want to MAXIMIZE distance to prior actions)
    """
    
    def __init__(
        self,
        number_of_assets: int,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
        optimizer: torch.optim = torch.optim.Adam,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        soft_update: bool = True,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        weight_decay: float = 0,
        gamma: float = 0.99,
        early_stopping: bool = True,
        patience: int = 5,
        min_delta: float = 0,
        # === DDES Parameters ===
        use_ddes: bool = False,
        ddes_alpha: float = 1.0,
        ddes_alpha_final: float = 0.0,
        ddes_scaling_method: str = "fixed",
        ddes_n_prior_samples: int = 16,
    ):
        self.number_of_assets = number_of_assets
        self.actor = actor
        self.critic = critic
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.soft_update = soft_update
        self.tau = tau
        self.risk_preference = risk_preference
        self.gamma = gamma
        self.early_stopper = EarlyStopper(patience, min_delta) if early_stopping else None

        # === DDES Parameters ===
        self.use_ddes = use_ddes
        self.ddes_alpha = ddes_alpha
        self.ddes_alpha_initial = ddes_alpha
        self.ddes_alpha_final = ddes_alpha_final
        self.ddes_scaling_method = ddes_scaling_method
        self.ddes_n_prior_samples = ddes_n_prior_samples

        # Optimizers
        self.actor_optimizer = optimizer(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optimizer(
            critic.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
        )

        # Target networks
        if soft_update:
            self.target_actor = deepcopy(actor)
            self.target_critic = deepcopy(critic)
            self._soft_update(self.target_actor, self.actor, tau=1.0)
            self._soft_update(self.target_critic, self.critic, tau=1.0)

    def _soft_update(self, target, source, tau):
        """Soft-update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - tau) * target_param.data + tau * source_param.data
            )

    def _compute_portfolio_returns(self, state, portfolio_allocation):
        """Compute portfolio returns for a batch.
        
        Args:
            state: Tensor of shape (batch, lookback, n_assets)
            portfolio_allocation: Tensor of shape (batch, n_assets)
        
        Returns:
            Tensor of shape (batch, lookback) with daily returns per sample
        """
        batch_size = state.size(0)
        state_3d = state.view(batch_size, -1, self.number_of_assets)
        alloc_expanded = portfolio_allocation.unsqueeze(1)
        portfolio_returns = torch.sum(state_3d * alloc_expanded, dim=-1)
        return portfolio_returns

    

    # =========================================================================
    # DDES-specific methods
    # =========================================================================
    
    def _compute_diversity_loss(self, current_state, current_action, replay_buffer):
        """Compute diversity relative to actions from SIMILAR states.
        
        Hong et al. (2018): diversity should be state-conditioned.
        """
        if len(replay_buffer) < self.ddes_n_prior_samples * 2:
            return torch.tensor(0.0)
        
        # Sample candidates (more than needed, then filter)
        n_candidates = min(len(replay_buffer), self.ddes_n_prior_samples * 5)
        candidates = replay_buffer.sample(n_candidates)
        
        # Extract states and actions: buffer format is (state, action, reward, next_state)
        candidate_states = torch.stack([t[0].flatten() for t in candidates])
        candidate_actions = torch.stack([t[1] for t in candidates])
        
        # Find k-nearest states by L2 distance
        current_flat = current_state.flatten()
        distances = torch.norm(candidate_states - current_flat, dim=-1)
        k = min(self.ddes_n_prior_samples, len(candidates))
        _, nearest_idx = torch.topk(distances, k, largest=False)
        
        prior_actions = candidate_actions[nearest_idx]
        
        # Compute diversity (negative = we maximize distance)
        if current_action.dim() == 1:
            current_action = current_action.unsqueeze(0)
        
        action_dist = torch.mean((current_action - prior_actions) ** 2, dim=-1)
        return -torch.mean(action_dist)

    def _update_ddes_alpha(self, epoch, total_epochs, current_diversity):
        """Update alpha based on scaling method.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            current_diversity: Current average diversity loss magnitude
            
        Returns:
            Updated alpha value
        """
        if self.ddes_scaling_method == "fixed":
            return self.ddes_alpha_initial
        
        elif self.ddes_scaling_method == "linear_decay":
            progress = epoch / total_epochs
            return self.ddes_alpha_initial + progress * (
                self.ddes_alpha_final - self.ddes_alpha_initial
            )
        
        elif self.ddes_scaling_method == "distance_based":
            # Scale alpha inversely with current diversity
            if current_diversity > 0:
                return self.ddes_alpha_initial / (1 + current_diversity)
            return self.ddes_alpha_initial
        
        return self.ddes_alpha_initial

    # =========================================================================
    # Training loop
    # =========================================================================

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        noise: float = 0.2,
        verbose: int = 1,
    ):
        """Training loop for DDPG with optional DDES.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of training epochs.
            noise: Exploration noise std (ignored if use_ddes=True).
            verbose: Verbosity level (0, 1, or 2).
        """
        replay_buffer = ReplayBuffer()
        current_diversity = 0.0  # Track for adaptive alpha

        for epoch in range(num_epochs):
            total_actor_loss = 0
            total_critic_loss = 0
            total_diversity_loss = 0

            # Update DDES alpha if enabled
            if self.use_ddes:
                self.ddes_alpha = self._update_ddes_alpha(
                    epoch, num_epochs, current_diversity
                )

            for state, next_state in train_loader:
                batch_size = state.size(0)

                # Flatten preserving batch dimension
                state_flat = state.flatten(start_dim=1)
                next_state_flat = next_state.flatten(start_dim=1)

                # Get current action from actor
                portfolio_allocation = self.actor(state_flat)

                # === EXPLORATION: DDES vs Gaussian noise ===
                if self.use_ddes:
                    # DDES: only some exploration noise - diversity term handles exploration
                    exploration_noise = torch.normal(0, noise * 0.5, portfolio_allocation.shape)
                    action_for_buffer = portfolio_allocation + exploration_noise
                else:
                    # Standard DDPG: Add Gaussian noise for exploration
                    exploration_noise = torch.normal(0, noise, portfolio_allocation.shape)
                    action_for_buffer = portfolio_allocation + exploration_noise

                # Compute reward using Sharpe ratio
                portfolio_returns = self._compute_portfolio_returns(next_state, action_for_buffer)
                reward = compute_reward(portfolio_returns, "sharpe")

                # Store transitions in replay buffer
                for i in range(batch_size):
                    replay_buffer.push((
                        state[i].detach(),
                        action_for_buffer[i].detach(),
                        reward[i].detach() if reward.dim() > 0 else reward.detach(),
                        next_state[i].detach()
                    ))

                # Sample transition from replay buffer
                transition = replay_buffer.sample(1)
                sampled_state = transition[0][0].unsqueeze(0)
                sampled_action = transition[0][1].unsqueeze(0)
                sampled_reward = transition[0][2]
                sampled_next_state = transition[0][3].unsqueeze(0)

                sampled_state_flat = sampled_state.flatten(start_dim=1)
                sampled_next_state_flat = sampled_next_state.flatten(start_dim=1)

                # Recompute action for sampled state
                sampled_portfolio_allocation = self.actor(sampled_state_flat)

                # Compute next Q-value using target networks
                if self.soft_update:
                    next_portfolio_allocation = self.target_actor(sampled_next_state_flat)
                    next_q_value = self.target_critic(
                        torch.cat((sampled_next_state_flat, next_portfolio_allocation), dim=1)
                    )
                else:
                    next_portfolio_allocation = self.actor(sampled_next_state_flat)
                    next_q_value = self.critic(
                        torch.cat((sampled_next_state_flat, next_portfolio_allocation), dim=1)
                    )

                # Compute target Q-value
                target_q_value = sampled_reward + self.gamma * next_q_value

                # === CRITIC UPDATE (same for both) ===
                q_value = self.critic(
                    torch.cat((sampled_state_flat, sampled_action), dim=1)
                )
                critic_loss = (target_q_value - q_value).pow(2).mean()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # === ACTOR UPDATE ===
                critic_input = torch.cat(
                    (sampled_state_flat, sampled_portfolio_allocation), dim=1
                )
                actor_q_loss = -self.critic(critic_input).mean()

                # === DDES: Add diversity term to actor loss ===
                if self.use_ddes:
                    diversity_loss = self._compute_diversity_loss(
                        sampled_state, sampled_portfolio_allocation, replay_buffer
                    )
                    actor_loss = actor_q_loss + self.ddes_alpha * diversity_loss
                    total_diversity_loss += diversity_loss.item()
                else:
                    actor_loss = actor_q_loss

                # Add L1/L2 regularization
                l1_actor = sum(w.abs().sum() for w in self.actor.parameters())
                l2_actor = sum(w.pow(2).sum() for w in self.actor.parameters())
                actor_loss = actor_loss + self.l1_lambda * l1_actor + self.l2_lambda * l2_actor

                # Backprop actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

            # Track average diversity for adaptive alpha
            if self.use_ddes:
                current_diversity = abs(total_diversity_loss / len(train_loader))

            # Average losses
            avg_actor_loss = total_actor_loss / len(train_loader)
            avg_critic_loss = total_critic_loss / len(train_loader)
            avg_diversity_loss = total_diversity_loss / len(train_loader) if self.use_ddes else 0

            # === VALIDATION & EARLY STOPPING ===
            if self.early_stopper:
                with torch.no_grad():
                    val_critic_loss = 0
                    for state, next_state in val_loader:
                        batch_size = state.size(0)
                        state_flat = state.flatten(start_dim=1)
                        next_state_flat = next_state.flatten(start_dim=1)

                        portfolio_allocation = self.actor(state_flat)
                        q_value = self.critic(
                            torch.cat((state_flat, portfolio_allocation), dim=1)
                        )

                        if self.soft_update:
                            next_portfolio_allocation = self.target_actor(next_state_flat)
                            next_q_value = self.target_critic(
                                torch.cat((next_state_flat, next_portfolio_allocation), dim=1)
                            )
                        else:
                            next_portfolio_allocation = self.actor(next_state_flat)
                            next_q_value = self.critic(
                                torch.cat((next_state_flat, next_portfolio_allocation), dim=1)
                            )

                        portfolio_returns = self._compute_portfolio_returns(
                            next_state, portfolio_allocation
                        )
                        reward = compute_reward(portfolio_returns, "sharpe")

                        if reward.dim() == 0:
                            reward = reward.unsqueeze(0)
                        target_q_value = reward.unsqueeze(1) + self.gamma * next_q_value
                        val_critic_loss += (target_q_value - q_value).pow(2).mean().item()

                    avg_val_critic_loss = val_critic_loss / len(val_loader)

                # Logging
                if verbose > 0:
                    if self.use_ddes:
                        print(f'Epoch {epoch+1}/{num_epochs}, '
                              f'Actor: {avg_actor_loss:.6f}, '
                              f'Critic: {avg_critic_loss:.6f}, '
                              f'Diversity: {avg_diversity_loss:.6f}, '
                              f'Alpha: {self.ddes_alpha:.4f}, '
                              f'Val: {avg_val_critic_loss:.6f}')
                    else:
                        print(f'Epoch {epoch+1}/{num_epochs}, '
                              f'Actor: {avg_actor_loss:.6f}, '
                              f'Critic: {avg_critic_loss:.6f}, '
                              f'Val: {avg_val_critic_loss:.6f}')

                if self.early_stopper.early_stop(avg_val_critic_loss, verbose=verbose):
                    break
            else:
                if verbose > 0:
                    if self.use_ddes:
                        print(f'Epoch {epoch+1}/{num_epochs}, '
                              f'Actor: {avg_actor_loss:.6f}, '
                              f'Critic: {avg_critic_loss:.6f}, '
                              f'Diversity: {avg_diversity_loss:.6f}, '
                              f'Alpha: {self.ddes_alpha:.4f}')
                    else:
                        print(f'Epoch {epoch+1}/{num_epochs}, '
                              f'Actor: {avg_actor_loss:.6f}, '
                              f'Critic: {avg_critic_loss:.6f}')

            # Soft update target networks
            if self.soft_update:
                self._soft_update(self.target_actor, self.actor, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)