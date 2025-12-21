# =============================================================================
# pga_map_elites.py
# =============================================================================
# PGA-MAP-Elites (Nilsson & Cully, GECCO 2021)
# https://github.com/ollenilsson19/PGA-MAP-Elites
# =============================================================================

import numpy as np
import torch
from sklearn.neighbors import KDTree
from functools import partial

from .official_networks import Actor, Critic
from .official_utils import ReplayBuffer, Individual, add_to_archive, cvt
from .official_variational_operators import VariationalOperator


def run(env, cfg):
    """
    Run PGA-MAP-Elites.
    
    Args:
        env: Gym-style environment
        cfg: Config dict with all parameters
    
    Returns:
        archive: Dict mapping niche -> Individual
    """
    # Unpack config
    pga = cfg['pga']
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0
    
    # Seeds
    torch.manual_seed(cfg['seeds'][0])
    np.random.seed(cfg['seeds'][0])
    
    print("="*60)
    print(f"PGA-MAP-Elites: {pga['max_evals']} evals, {pga['n_niches']} niches")
    print("="*60)
    
    # =========================================================================
    # Initialize
    # =========================================================================
    
    # Actor factory
    actor_fn = partial(
        Actor,
        state_dim,
        action_dim,
        max_action,
        cfg['hidden_sizes'],
    )
    
    # Critic (TD3)
    critic = Critic(
        state_dim,
        action_dim,
        max_action,
        discount=cfg['gamma'],
        tau=cfg['tau'],
        policy_noise=pga['policy_noise'] * max_action,
        noise_clip=pga['noise_clip'] * max_action,
        policy_freq=pga['policy_freq'],
    )
    
    # Replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Variation operator
    var_op = VariationalOperator(
        actor_fn=actor_fn,
        num_cpu=1,
        gradient_op=True,
        crossover_op="iso_dd",
        mutation_op=None,
        learning_rate=cfg['actor_lr'],
        iso_sigma=pga['iso_sigma'],
        line_sigma=pga['line_sigma'],
    )
    
    # CVT + KDTree
    c = cvt(pga['n_niches'], dim=2, samples=25000)
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    
    # Archive
    archive = {}
    n_evals = 0
    
    # =========================================================================
    # Main loop
    # =========================================================================
    
    while n_evals < pga['max_evals']:
        to_evaluate = []
        
        # Random init or Variation
        if n_evals < pga['random_init']:
            for _ in range(pga['batch_size']):
                to_evaluate.append(actor_fn())
        else:
            # Train critic
            if replay_buffer.size > pga['train_batch_size']:
                critic.train(
                    archive,
                    replay_buffer,
                    pga['nr_of_steps_crit'],
                    batch_size=pga['train_batch_size']
                )
                states = replay_buffer.sample_state(
                    pga['train_batch_size'],
                    pga['nr_of_steps_act']
                )
            else:
                states = None
            
            # Variation
            to_evaluate = var_op(
                archive,
                pga['batch_size'],
                pga['proportion_evo'],
                critic=critic,
                states=states,
                train_batch_size=pga['train_batch_size'],
                nr_of_steps_act=pga['nr_of_steps_act']
            )
        
        # Evaluate
        added = 0
        for actor in to_evaluate:
            fitness, desc, transitions = eval_policy(actor, env)
            
            # Store transitions
            if transitions:
                s, a, ns, r, nd = zip(*transitions)
                replay_buffer.add((
                    np.array(s), np.array(a), np.array(ns),
                    np.array(r).reshape(-1,1), np.array(nd).reshape(-1,1)
                ))
            
            # Add to archive
            ind = Individual(actor, desc, fitness)
            if add_to_archive(ind, desc, archive, kdt):
                added += 1
        
        n_evals += len(to_evaluate)
        
        # Log
        if n_evals % 500 == 0 or n_evals >= pga['max_evals']:
            cov = 100 * len(archive) / pga['n_niches']
            best = max((x.fitness for x in archive.values()), default=0)
            print(f"[{n_evals:5d}] Archive: {len(archive):3d} ({cov:.0f}%) | "
                  f"Best: {best:.4f} | +{added}")
    
    var_op.close()
    return archive


def eval_policy(actor, env):
    """Evaluate actor, return (fitness, behavior_desc, transitions)."""
    state = env.reset()
    transitions = []
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action = actor.select_action(state)
        
        next_state, reward, done, info = env.step(action)
        transitions.append((state.copy(), action.copy(), next_state.copy(), 
                           reward, 0.0 if done else 1.0))
        total_reward += reward
        state = next_state
    
    desc = np.array([info.get('volatility', 0.0), 
                     info.get('diversification', 0.0)])
    
    return total_reward, desc, transitions

"""
Policy Evaluation for PGA-MAP-Elites

Aligned with original implementation from:
https://github.com/ollenilsson19/PGA-MAP-Elites/blob/master/vectorized_env.py

Every design choice traced via [QDGYM] or [FINRL] tags.
"""

import numpy as np
import torch


def eval_policy(actor, env):
    """
    [QDGYM] Evaluate policy in environment.
    
    Follows original PGA-MAP-Elites evaluation pattern from vectorized_env.py.
    Uses env.tot_reward and env.desc instead of computing manually.
    
    Args:
        actor: Policy network with select_action(state) method [QDGYM]
        env: PortfolioEnv with QDgym interface [QDGYM + FINRL]
        
    Returns:
        tuple: (fitness, behavior_descriptor, transitions)
            - fitness: float, env.tot_reward [QDGYM]
            - behavior_descriptor: np.array shape (2,), env.desc [QDGYM]
            - transitions: list of (s, a, s', r, done_bool) [QDGYM]
    """
    # [QDGYM] Reset environment - returns state only
    state = env.reset()
    
    transitions = []
    done = False
    
    # [QDGYM] Main evaluation loop
    while not done:
        # [QDGYM] Get action from policy (no gradient needed)
        with torch.no_grad():
            action = actor.select_action(state)
        
        # [QDGYM] Step environment
        next_state, reward, done, info = env.step(action)
        
        # -----------------------------------------------------------------
        # [QDGYM] Critical: done_bool calculation from vectorized_env.py
        # 
        # From original code:
        #   done_bool = float(done) if env.T < env._max_episode_steps else 0
        #
        # This is for TD3 bootstrapping:
        # - If episode ended naturally (T < max): done_bool = 1.0 (no bootstrap)
        # - If episode truncated (T >= max): done_bool = 0.0 (bootstrap)
        # -----------------------------------------------------------------
        if env.T < env._max_episode_steps:
            done_bool = float(done)
        else:
            done_bool = 0.0  # [QDGYM] Truncation, not termination
        
        # [QDGYM] Store transition for replay buffer
        transitions.append((
            state.copy(),
            action.copy(),
            next_state.copy(),
            reward,
            done_bool
        ))
        
        state = next_state
    
    # -----------------------------------------------------------------
    # [QDGYM] Return values from env attributes, not manual computation
    # This ensures consistency with QDgym evaluation pattern
    # -----------------------------------------------------------------
    fitness = env.tot_reward      # [QDGYM] Cumulative reward
    descriptor = env.desc.copy()  # [QDGYM] Behavior descriptor
    
    return fitness, descriptor, transitions


def format_transitions_for_buffer(transitions_list):
    """
    [QDGYM] Format transitions for ReplayBuffer.add()
    
    Original PGA-MAP-Elites ReplayBuffer expects:
        transitions = (states, actions, next_states, rewards, not_dones)
    Each is a numpy array where first dim is batch.
    
    Args:
        transitions_list: List of (s, a, s', r, done_bool) tuples
        
    Returns:
        tuple of arrays: (states, actions, next_states, rewards, not_dones)
    """
    if not transitions_list:
        return None
    
    states = np.array([t[0] for t in transitions_list])
    actions = np.array([t[1] for t in transitions_list])
    next_states = np.array([t[2] for t in transitions_list])
    rewards = np.array([t[3] for t in transitions_list]).reshape(-1, 1)
    # [QDGYM] not_done = 1 - done_bool
    not_dones = 1.0 - np.array([t[4] for t in transitions_list]).reshape(-1, 1)
    
    return (states, actions, next_states, rewards, not_dones)


if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.path.insert(0, '/home/claude')
    from portfolio_env import PortfolioEnv
    
    # Mock actor for testing
    class MockActor:
        def __init__(self, action_dim):
            self.action_dim = action_dim
        
        def select_action(self, state):
            return np.random.randn(self.action_dim)
    
    # Create environment
    np.random.seed(42)
    returns = np.random.randn(100, 5) * 0.02
    df = pd.DataFrame(returns)
    env = PortfolioEnv(df, lookback=10, episode_len=20)
    
    # Create actor
    actor = MockActor(env.action_dim)
    
    # Evaluate
    fitness, desc, transitions = eval_policy(actor, env)
    
    print("=== eval_policy Test ===")
    print(f"Fitness (env.tot_reward): {fitness:.6f}")
    print(f"Descriptor (env.desc): {desc}")
    print(f"Num transitions: {len(transitions)}")
    print(f"Last 3 done_bool values: {[t[4] for t in transitions[-3:]]}")
    
    # Test buffer formatting
    formatted = format_transitions_for_buffer(transitions)
    print(f"\n=== Buffer Format ===")
    print(f"States: {formatted[0].shape}, Actions: {formatted[1].shape}")
    print(f"Not_dones (last 3): {formatted[4][-3:].flatten()}")