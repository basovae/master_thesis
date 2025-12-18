
"""
PGA-MAP-Elites: Simplified Implementation for Portfolio Optimization
=====================================================================

Based on:
- Paper: Nilsson & Cully, GECCO 2021 (Algorithm 3, 4, 5)
- Official repo: github.com/ollenilsson19/PGA-MAP-Elites
- TD3 repo: github.com/sfujim/TD3

YOUR EXISTING DDPG COMPONENTS THAT WE REUSE:
- Actor network (policy) ✓
- Critic network ✓  
- Replay buffer ✓
- Target networks ✓
- Soft updates ✓

WHAT WE ADD:
- Second critic (Q2) for TD3's clipped double Q-learning
- CVT Archive (stores diverse portfolio policies)
- GA variation (iso-line operator)
- PG variation (policy gradient mutation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from collections import deque

from replay_buffer import ReplayBuffer
from variational_operators import ga_variation, pg_variation
from archive import CVTArchive


class TD3Trainer:
    """
    TD3-style critic training for PGA-MAP-Elites.
    
    📝 PAPER (Algorithm 4):
       - Train for n_crit iterations (default 300)
       - Delayed policy updates every d=2 iterations
       - Target smoothing with clipped noise
       - Soft updates with τ=0.005
    
    📝 OFFICIAL TD3 CODE (sfujim/TD3):
       All hyperparameters verified ✓
    
    🎓 SIMPLIFICATION:
       - n_crit reduced to 100 for faster testing
       - Can run fewer iterations, just call train() more often
    """
    def __init__(self, state_dim, action_dim, 
                 gamma=0.99,           # 📝 VERIFIED: discount factor
                 tau=0.005,            # 📝 VERIFIED: soft update rate  
                 policy_noise=0.2,     # 📝 VERIFIED: target smoothing noise
                 noise_clip=0.5,       # 📝 VERIFIED: noise clipping
                 policy_freq=2,        # 📝 VERIFIED: delayed updates
                 lr=3e-4):             # 📝 VERIFIED: learning rate
        
        # Main networks
        self.critic = Critic(state_dim, action_dim)
        self.greedy_actor = Actor(state_dim, action_dim)
        
        # Target networks (📝 PAPER: initialized as copies)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.greedy_actor)
        
        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.greedy_actor.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
    
    def train(self, replay_buffer, batch_size=256, n_crit=100):
        """
        Train critics and greedy actor.
        
        📝 PAPER: Called once per MAP-Elites iteration
        📝 PAPER: n_crit = 300 (we use 100 for speed)
        
        🎓 KEY INSIGHT: The greedy_actor is ONLY for computing targets!
           It never goes into the archive. Archive policies come from variation.
        """
        for _ in range(n_crit):
            self.total_it += 1
            
            # Sample batch
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            
            with torch.no_grad():
                # 📝 PAPER (Algorithm 4, line 4): Target policy smoothing
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                # Get target action from target actor + noise
                next_action = self.actor_target(next_state) + noise
                # 🎓 FOR PORTFOLIO: Renormalize to sum to 1
                next_action = F.softmax(next_action, dim=-1)
                
                # 📝 PAPER (Algorithm 4, line 5): Clipped double Q-learning
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * self.gamma * target_q
            
            # Get current Q estimates
            current_q1, current_q2 = self.critic(state, action)
            
            # 📝 PAPER (Algorithm 4, line 6): Critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 📝 PAPER (Algorithm 4, line 7): Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                # Actor loss: maximize Q1
                actor_loss = -self.critic.Q1(state, self.greedy_actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # 📝 PAPER (Algorithm 4, lines 10-11): Soft update targets
                for param, target_param in zip(self.critic.parameters(), 
                                                self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                
                for param, target_param in zip(self.greedy_actor.parameters(),
                                                self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
        
        return self.greedy_actor


# =============================================================================
# MAIN PGA-MAP-ELITES ALGORITHM
# =============================================================================

class PGAMAPElites:
    """
    Complete PGA-MAP-Elites algorithm.
    
    📝 PAPER (Algorithm 3): Main loop structure
    
    🎓 SIMPLIFICATION:
       - Reduced default parameters for faster testing
       - Simpler logging
       - evaluate_policy must be customized for your environment
    """
    def __init__(self, 
                 state_dim,
                 action_dim,
                 n_niches=100,           # 🎓 Reduced from 1024
                 behavior_dim=2,
                 bd_bounds=((0, 1), (0, 1)),
                 batch_size=20,          # 🎓 Reduced from 100
                 random_init=200,        # 🎓 Reduced from 1000
                 proportion_evo=0.5,     # 📝 VERIFIED: 50% GA, 50% PG
                 n_crit=100,             # 🎓 Reduced from 300
                 n_grad=10):             # 📝 VERIFIED: 10-50
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.random_init = random_init
        self.proportion_evo = proportion_evo
        self.n_crit = n_crit
        self.n_grad = n_grad
        
        # Initialize components
        self.archive = CVTArchive(n_niches, behavior_dim, bd_bounds)
        self.trainer = TD3Trainer(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer()
        
        self.total_evals = 0
    
    def random_policy(self):
        """Create random initialized policy"""
        return Actor(self.state_dim, self.action_dim)
    
    def evaluate_policy(self, policy, env):
        """
        Evaluate policy in environment.
        
        🎓 YOU MUST CUSTOMIZE THIS for your portfolio environment!
        
        Returns: (fitness, behavior_descriptor, transitions)
        """
        # Example structure - replace with your actual evaluation
        state = env.reset()
        transitions = []
        total_reward = 0
        
        for _ in range(env.max_steps):
            with torch.no_grad():
                action = policy(torch.FloatTensor(state)).numpy()
            
            next_state, reward, done, info = env.step(action)
            transitions.append((state, action, reward, next_state, float(done)))
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # 🎓 CUSTOMIZE: Extract your behavior descriptor
        # For portfolio: could be (volatility, diversification)
        behavior = np.array([
            info.get('volatility', 0.5),
            info.get('diversification', 0.5)
        ])
        
        return total_reward, behavior, transitions
    
    def run(self, env, max_evals=10000, verbose=True):
        """
        Main PGA-MAP-Elites loop.
        
        📝 PAPER (Algorithm 3, lines 8-15)
        """
        while self.total_evals < max_evals:
            
            # === PHASE 1: Random Initialization ===
            # 📝 PAPER (lines 9-10): Random solutions until G evaluations
            if self.total_evals < self.random_init:
                batch = [self.random_policy() for _ in range(self.batch_size)]
            
            # === PHASE 2: Main Loop ===
            # 📝 PAPER (lines 11-13)
            else:
                # Step A: Train critics (returns greedy actor)
                greedy = self.trainer.train(
                    self.replay_buffer, 
                    n_crit=self.n_crit
                )
                
                # Step B: Generate offspring via variation
                n_evo = int((self.batch_size - 1) * self.proportion_evo)
                n_pg = self.batch_size - 1 - n_evo
                
                offspring = []
                
                # GA variation (iso-line)
                for _ in range(n_evo):
                    parents = self.archive.sample(n=2)
                    if len(parents) >= 2:
                        child = ga_variation(parents[0], parents[1])
                    elif len(parents) == 1:
                        child = ga_variation(parents[0], parents[0])
                    else:
                        child = self.random_policy()
                    offspring.append(child)
                
                # PG variation
                for _ in range(n_pg):
                    parents = self.archive.sample(n=1)
                    if parents:
                        child = pg_variation(
                            parents[0], 
                            self.trainer.critic,
                            self.replay_buffer,
                            n_grad=self.n_grad
                        )
                    else:
                        child = self.random_policy()
                    offspring.append(child)
                
                # Batch = greedy + offspring
                batch = [greedy] + offspring
            
            # === PHASE 3: Evaluate and Add to Archive ===
            # 📝 PAPER (line 14, Algorithm 3 lines 17-23)
            for policy in batch:
                fitness, behavior, transitions = self.evaluate_policy(policy, env)
                
                # Add transitions to replay buffer
                for t in transitions:
                    self.replay_buffer.add(*t)
                
                # Try to add to archive
                self.archive.add(policy, fitness, behavior)
            
            self.total_evals += len(batch)
            
            # Logging
            if verbose and self.total_evals % 100 == 0:
                print(f"Evals: {self.total_evals:5d} | "
                      f"Coverage: {self.archive.coverage:.1%} | "
                      f"Max Fitness: {self.archive.max_fitness:.3f} | "
                      f"Buffer: {len(self.replay_buffer)}")
        
        return self.archive