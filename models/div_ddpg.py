import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import predictors
from typing import Type

from copy import deepcopy

from utilities.data_processing import RLDataLoader
from utilities.metrics import RLEvaluator
from utilities.training import EarlyStopper, ReplayBuffer
from utilities.rewards import compute_reward


class DivDDPG:
    '''DDPG with Diversity-Driven Exploration Strategy (Hong et al. 2018).
    
    Adds a diversity term to the actor loss that encourages the policy
    to produce actions different from prior actions in the replay buffer.
    '''
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
        # Diversity parameters
        alpha: float = 1.0,
        alpha_final: float = 0.0,
        scaling_method: str = "distance_based",
        n_prior_samples: int = 16,
    ):
        '''Trains the Div-DDPG model.'''
        self.val_data = val_data
        dataloader = RLDataLoader(train_data, val_data, shuffle=False)

        self.number_of_assets = dataloader.number_of_assets
        number_of_datapoints = self.lookback_window + self.forecast_window
        self.input_size = self.number_of_assets * number_of_datapoints
        self.output_size = self.number_of_assets

        train_loader, val_loader = dataloader(
            batch_size=self.batch_size,
            window_size=self.lookback_window,
            forecast_size=self.forecast_window,
        )

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

        trainer = DivDDPGTrainer(
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
            # Diversity params
            alpha=alpha,
            alpha_final=alpha_final,
            scaling_method=scaling_method,
            n_prior_samples=n_prior_samples,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=self.verbose,
            num_epochs=num_epochs,
        )

        return self

    def evaluate(self, test_data: pd.DataFrame, dpo: bool = True) -> tuple:
        '''Evaluates the Div-DDPG model.'''
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


class DivDDPGTrainer:
    '''Trainer for DDPG with Diversity-Driven Exploration.
    
    The key modification from standard DDPG is the diversity term in actor loss:
        L_actor = -Q(s, a) + α * D(a, a_prior)
    
    Where D is MSE distance and a_prior are actions sampled from replay buffer.
    '''
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
        # Diversity parameters
        alpha: float = 1.0,
        alpha_final: float = 0.0,
        scaling_method: str = "distance_based",
        n_prior_samples: int = 16,
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

        # Diversity parameters
        self.alpha = alpha
        self.alpha_initial = alpha
        self.alpha_final = alpha_final
        self.scaling_method = scaling_method
        self.n_prior_samples = n_prior_samples

        self.actor_optimizer = optimizer(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optimizer(
            critic.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
        )

        if soft_update:
            self.target_actor = deepcopy(actor)
            self.target_critic = deepcopy(critic)
            self._soft_update(self.target_actor, self.actor, tau=1.0)
            self._soft_update(self.target_critic, self.critic, tau=1.0)

    def _soft_update(self, target, source, tau):
        '''Soft-update target network parameters.'''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)

    def _compute_diversity_loss(self, current_action, replay_buffer):
        '''Compute diversity loss as mean MSE to prior actions.
        
        From Hong et al. 2018: encourages current action to differ from
        actions stored in replay buffer.
        '''
        if len(replay_buffer) < self.n_prior_samples:
            return torch.tensor(0.0)
        
        # Sample prior actions from buffer
        prior_transitions = replay_buffer.sample(self.n_prior_samples)
        prior_actions = torch.stack([t[1] for t in prior_transitions])
        
        # MSE distance to each prior action
        distances = torch.mean((current_action - prior_actions) ** 2, dim=-1)
        
        # Return negative mean distance (we want to maximize distance)
        return -torch.mean(distances)

    def _update_alpha(self, epoch, total_epochs, current_diversity):
        '''Update alpha based on scaling method.'''
        if self.scaling_method == "fixed":
            return self.alpha_initial
        
        elif self.scaling_method == "linear_decay":
            # Linear decay from alpha_initial to alpha_final
            progress = epoch / total_epochs
            return self.alpha_initial + progress * (self.alpha_final - self.alpha_initial)
        
        elif self.scaling_method == "distance_based":
            # Scale alpha inversely with current diversity
            # High diversity -> low alpha, Low diversity -> high alpha
            if current_diversity > 0:
                return self.alpha_initial / (1 + current_diversity)
            return self.alpha_initial
        
        return self.alpha_initial

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        verbose: int = 1,
    ):
        '''Training loop for Div-DDPG.
        
        Key difference from DDPG: no exploration noise added to actions.
        Diversity term in loss handles exploration instead.
        '''
        replay_buffer = ReplayBuffer()
        current_diversity = 0.0

        for epoch in range(num_epochs):
            total_actor_loss = 0
            total_critic_loss = 0
            total_diversity_loss = 0

            # Update alpha based on scaling method
            self.alpha = self._update_alpha(epoch, num_epochs, current_diversity)

            for state, next_state in train_loader:
                # Get current action (no exploration noise - diversity handles it)
                portfolio_allocation = self.actor(state.flatten())

                # Compute reward
                portfolio_returns = torch.sum(
                    state.view(-1, self.number_of_assets) * portfolio_allocation,
                    dim=-1
                )
                reward = compute_reward(portfolio_returns, "sharpe")

                # Store transition
                replay_buffer.push((
                    state.detach(),
                    portfolio_allocation.detach(),
                    reward.detach(),
                    next_state.detach()
                ))

                # Sample from replay buffer
                transition = replay_buffer.sample(1)
                state = transition[0][0]
                sampled_action = transition[0][1]
                reward = transition[0][2]
                next_state = transition[0][3]

                # Current action for this state
                portfolio_allocation = self.actor(state.flatten())

                # Next state value (target networks if soft update)
                if self.soft_update:
                    next_action = self.target_actor(next_state.flatten())
                    next_q_value = self.target_critic(
                        torch.cat((next_state.flatten(), next_action.flatten()))
                    )
                else:
                    next_action = self.actor(next_state.flatten())
                    next_q_value = self.critic(
                        torch.cat((next_state.flatten(), next_action.flatten()))
                    )

                # Critic update
                target_q_value = reward + self.gamma * next_q_value
                q_value = self.critic(
                    torch.cat((state.flatten(), sampled_action.flatten()))
                )
                critic_loss = (target_q_value - q_value).pow(2)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # Actor update with diversity term
                critic_input = torch.cat((state.flatten(), portfolio_allocation.flatten()))
                actor_q_loss = -self.critic(critic_input)
                
                # Diversity loss
                diversity_loss = self._compute_diversity_loss(portfolio_allocation, replay_buffer)
                
                # Combined actor loss
                actor_loss = actor_q_loss + self.alpha * diversity_loss

                # L1/L2 regularization
                l1_actor = sum(w.abs().sum() for w in self.actor.parameters())
                l2_actor = sum(w.pow(2).sum() for w in self.actor.parameters())
                actor_loss += self.l1_lambda * l1_actor + self.l2_lambda * l2_actor

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_diversity_loss += diversity_loss.item()

            # Track average diversity for adaptive scaling
            current_diversity = abs(total_diversity_loss / len(train_loader))

            avg_actor_loss = total_actor_loss / len(train_loader)
            avg_critic_loss = total_critic_loss / len(train_loader)
            avg_diversity_loss = total_diversity_loss / len(train_loader)

            # Validation and early stopping
            if self.early_stopper:
                with torch.no_grad():
                    val_critic_loss = 0
                    for state, next_state in val_loader:
                        portfolio_allocation = self.actor(state.flatten())
                        q_value = self.critic(
                            torch.cat((state.flatten(), portfolio_allocation.flatten()))
                        )

                        if self.soft_update:
                            next_action = self.target_actor(next_state.flatten())
                            next_q_value = self.target_critic(
                                torch.cat((next_state.flatten(), next_action.flatten()))
                            )
                        else:
                            next_action = self.actor(next_state.flatten())
                            next_q_value = self.critic(
                                torch.cat((next_state.flatten(), next_action.flatten()))
                            )

                        avg_profit = torch.mean(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation, dim=-1)
                        ).detach().cpu()
                        volatility = torch.std(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation, dim=-1),
                            correction=0,
                        ).detach().cpu()
                        reward = avg_profit + self.risk_preference * volatility

                        target_q_value = reward + self.gamma * next_q_value
                        val_critic_loss += (target_q_value - q_value).pow(2).item()

                    avg_val_critic_loss = val_critic_loss / len(val_loader)

                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, '
                          f'Actor: {avg_actor_loss:.6f}, '
                          f'Critic: {avg_critic_loss:.6f}, '
                          f'Diversity: {avg_diversity_loss:.6f}, '
                          f'Alpha: {self.alpha:.4f}, '
                          f'Val: {avg_val_critic_loss:.6f}')

                if self.early_stopper.early_stop(avg_val_critic_loss, verbose=verbose):
                    break
            else:
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, '
                          f'Actor: {avg_actor_loss:.6f}, '
                          f'Critic: {avg_critic_loss:.6f}, '
                          f'Diversity: {avg_diversity_loss:.6f}, '
                          f'Alpha: {self.alpha:.4f}')

            # Soft update target networks
            if self.soft_update:
                self._soft_update(self.target_actor, self.actor, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)