# =============================================================================
# pga_map_elites.py
# =============================================================================
# PGA-MAP-Elites (Nilsson & Cully, 2021)
# Simplified for portfolio optimization
# =============================================================================

import numpy as np
import torch
from copy import deepcopy

from .networks import Actor, Critic
from .official_logic import ReplayBuffer, Individual, add_to_archive, cvt
from .variational_operators import VariationalOperator


def run_pga(env, cfg):
    """
    Run PGA-MAP-Elites.
    
    Args:
        env: Gym-style environment
        cfg: Unified config dict (uses cfg['pga'] for PGA-specific params)
    
    Returns:
        archive: Dict mapping niche_id → Individual
    """
    pga = cfg['pga']
    seed = cfg['seeds'][0]
    
    # Dimensions from environment
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # =========================================================================
    # Initialize components
    # =========================================================================
    
    # CVT centroids for behavior space
    bd_bounds = (pga['bd_min'], pga['bd_max'])
    centroids, kdt = cvt(pga['n_niches'], dim=2, samples=25000, bd_bounds=bd_bounds)
    
    # Empty archive
    archive = {}
    
    # Critic (TD3-style)
    critic = Critic(
        state_dim, action_dim, max_action=1.0,
        discount=cfg['gamma'],
        tau=cfg['tau'],
        policy_noise=pga['policy_noise'],
        noise_clip=pga['noise_clip'],
        policy_freq=pga['policy_freq'],
    )
    
    # Replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Variation operator
    def make_actor():
        return Actor(state_dim, action_dim, max_action=1.0, 
                     neurons_list=cfg['hidden_sizes'])
    
    var_op = VariationalOperator(
        actor_fn=make_actor,
        iso_sigma=pga['iso_sigma'],
        line_sigma=pga['line_sigma'],
        learning_rate=cfg['actor_lr'],
    )
    
    # =========================================================================
    # Main loop
    # =========================================================================
    n_evals = 0
    iteration = 0
    
    print(f"PGA-MAP-Elites: {pga['max_evals']} evals, {pga['n_niches']} niches")
    
    while n_evals < pga['max_evals']:
        iteration += 1
        
        # ---------------------------------------------------------------------
        # Generate offspring
        # ---------------------------------------------------------------------
        if n_evals < pga['random_init']:
            # Random initialization
            offspring = [make_actor() for _ in range(pga['batch_size'])]
            for actor in offspring:
                actor.type = "random"
        else:
            # Variation (GA + PG)
            states = replay_buffer.sample_state(pga['train_batch_size'], pga['nr_of_steps_act'])
            offspring = var_op(
                archive=archive,
                batch_size=pga['batch_size'],
                proportion_evo=pga['proportion_evo'],
                critic=critic,
                states=states,
                nr_of_steps_act=pga['nr_of_steps_act'],
            )
        
        # ---------------------------------------------------------------------
        # Evaluate and add to archive
        # ---------------------------------------------------------------------
        added = 0
        for actor in offspring:
            fitness, behavior, transitions = evaluate(actor, env)
            
            # Store transitions
            for t in transitions:
                replay_buffer.add(*t)
            
            # Normalize behavior to [0,1]
            behavior_norm = normalize_bd(behavior, bd_bounds)
            
            # Try add to archive
            ind = Individual(actor, behavior_norm, fitness)
            if add_to_archive(ind, archive, kdt):
                added += 1
        
        n_evals += len(offspring)
        
        # ---------------------------------------------------------------------
        # Train critic
        # ---------------------------------------------------------------------
        if len(replay_buffer) > 1000:
            critic.train(archive, replay_buffer, pga['nr_of_steps_crit'],
                         batch_size=pga['train_batch_size'])
        
        # ---------------------------------------------------------------------
        # Log progress
        # ---------------------------------------------------------------------
        if iteration % 10 == 0:
            coverage = 100 * len(archive) / pga['n_niches']
            best = max((ind.fitness for ind in archive.values()), default=0)
            print(f"Iter {iteration:3d} | Evals {n_evals:5d} | "
                  f"Archive {len(archive):3d} ({coverage:.1f}%) | "
                  f"Best {best:.4f} | +{added}")
    
    var_op.close()
    return archive


# =============================================================================
# Helpers
# =============================================================================

def evaluate(actor, env):
    """Evaluate actor, return (fitness, behavior, transitions)."""
    state = env.reset()
    transitions = []
    total_reward = 0
    
    done = False
    while not done:
        with torch.no_grad():
            action = actor(torch.FloatTensor(state).unsqueeze(0)).cpu().numpy().flatten()
        
        next_state, reward, done, info = env.step(action)
        transitions.append((state, action, next_state, reward, float(not done)))
        total_reward += reward
        state = next_state
    
    behavior = np.array([
        info.get('volatility', 0.0),
        info.get('diversification', 0.0)
    ])
    
    return total_reward, behavior, transitions


def normalize_bd(behavior, bd_bounds):
    """Normalize behavior descriptor to [0,1]."""
    bd_min, bd_max = bd_bounds
    return np.clip((behavior - bd_min) / (np.array(bd_max) - bd_min + 1e-8), 0, 1)