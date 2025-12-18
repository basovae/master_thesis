import copy

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Critic, Actor
from .archive import CVTArchive
from .replay_buffer import ReplayBuffer
from .variational_operators import variation


class TD3Trainer:
    """
    TD3-style critic training for PGA-MAP-Elites
    Called once per MAP-Elites iteration (Algorithm 3, line 12)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,           # γ
        tau=0.005,               # Target update rate
        policy_noise=0.2,        # σ_p (target smoothing)
        noise_clip=0.5,          # c (clipping range)
        policy_freq=2,           # d (delayed updates)
        lr=3e-4                  # Learning rate
    ):
        # Main networks
        self.critic = Critic(state_dim, action_dim)
        self.greedy_actor = Actor(state_dim, action_dim, max_action)
        
        # Target networks (initialized as copies)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.greedy_actor)
        
        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.greedy_actor.parameters(), lr=lr)
        
        # Hyperparameters
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        
        self.total_it = 0
    
    def train(self, replay_buffer, batch_size=256, n_crit=300):
        """
        Algorithm 4: TRAIN_CRITIC procedure
        
        Args:
            replay_buffer: Experience buffer B
            batch_size: N = 256
            n_crit: Number of training iterations (300 default)
        
        Returns:
            greedy_actor: Current greedy controller π_φc
        """
        for _ in range(n_crit):
            self.total_it += 1
            
            # Sample batch (line 3)
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            
            with torch.no_grad():
                # Target policy smoothing (line 4)
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                # Compute target action with noise (line 5)
                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)
                
                # Clipped double Q-learning: min of two targets (line 5)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.discount * target_Q
            
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Critic loss (line 6)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Update critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Delayed policy updates (line 7)
            if self.total_it % self.policy_freq == 0:
                # Actor loss: maximize Q1 (line 8-9)
                # ∇_φ J(φ) = (1/N) Σ ∇_φ π_φc(s) ∇_a Q_θ1(s, a)|_{a=π_φc(s)}
                actor_loss = -self.critic.Q1(state, self.greedy_actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Soft update targets (lines 10-11)
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                
                for param, target_param in zip(
                    self.greedy_actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
        
        return self.greedy_actor
    
class PGAMAPElites:
    """
    Complete PGA-MAP-Elites implementation
    Based on Algorithm 3 in the GECCO 2021 paper
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        n_niches=1024,
        behavior_dim=2,
        eval_batch_size=100,         # b
        random_init=1000,            # G
        max_evals=1000000,           # I
        proportion_evo=0.5,
        n_crit=300,
        n_grad=10
    ):
        # Archive
        self.archive = CVTArchive(n_niches, behavior_dim)
        
        # TD3 Trainer (critics + greedy controller)
        self.trainer = TD3Trainer(state_dim, action_dim, max_action)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        
        # Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.eval_batch_size = eval_batch_size
        self.random_init = random_init
        self.max_evals = max_evals
        self.proportion_evo = proportion_evo
        self.n_crit = n_crit
        self.n_grad = n_grad
        
        self.total_evals = 0
    
    def evaluate_policy(self, policy, env, max_steps=1000):
        """
        Algorithm 3, line 19: evaluate(π_φ)
        Returns (fitness, behavior_descriptor, transitions)
        """
        state = env.reset()
        transitions = []
        total_reward = 0
        
        for _ in range(max_steps):
            action = policy(torch.FloatTensor(state)).detach().numpy()
            next_state, reward, done, info = env.step(action)
            
            transitions.append((state, action, next_state, reward, float(done)))
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Compute behavior descriptor (domain-specific)
        behavior_descriptor = self.compute_bd(transitions, info)
        
        return total_reward, behavior_descriptor, transitions
    
    def compute_bd(self, transitions, info):
        """
        Compute behavior descriptor from episode.
        
        ⚠️ DEVIATION: Must be customized for portfolio optimization.
        """
        # Example: extract from environment info
        # For portfolio: volatility, diversification, turnover, etc.
        return np.array(info.get('behavior_descriptor', [0.0, 0.0]))
    
    def random_policy(self):
        """Generate random policy for initialization"""
        policy = Actor(self.state_dim, self.action_dim, self.max_action)
        return policy
    
    def run(self, env):
        """
        Main loop (Algorithm 3, lines 8-15)
        """
        while self.total_evals < self.max_evals:
            
            # Initialization phase (lines 9-10)
            if self.total_evals < self.random_init:
                batch = [self.random_policy() for _ in range(self.eval_batch_size)]
            
            # Main phase (lines 11-13)
            else:
                # Train critic and get greedy controller (line 12)
                greedy_controller = self.trainer.train(
                    self.replay_buffer, 
                    n_crit=self.n_crit
                )
                
                # Generate offspring via variation (line 13)
                offspring = variation(
                    batch_size=self.eval_batch_size - 1,
                    archive=self.archive,
                    critic=self.trainer.critic,
                    replay_buffer=self.replay_buffer,
                    proportion_evo=self.proportion_evo,
                    n_grad=self.n_grad
                )
                
                # Batch = greedy controller + offspring
                batch = [greedy_controller] + offspring
            
            # Evaluate and add to archive (line 14)
            for policy in batch:
                fitness, bd, transitions = self.evaluate_policy(policy, env)
                
                # Add transitions to replay buffer (line 20)
                self.replay_buffer.add_batch(transitions)
                
                # Add to archive (lines 21-23)
                self.archive.add_to_archive(policy, fitness, bd)
            
            self.total_evals += len(batch)
            
            # Logging
            if self.total_evals % 1000 == 0:
                coverage = sum(1 for p in self.archive.policies if p is not None)
                max_fit = np.max(self.archive.fitnesses[self.archive.fitnesses > -np.inf])
                print(f"Evals: {self.total_evals}, Coverage: {coverage}, Max Fitness: {max_fit:.2f}")
        
        return self.archive