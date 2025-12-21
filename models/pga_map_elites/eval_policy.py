"""
Evaluation logic for PGA-MAP-Elites

Simplified from original parallel implementation:
https://github.com/ollenilsson19/PGA-MAP-Elites

Original uses multiprocessing (parallel_worker, parallel_critic).
This version runs single-process for simplicity.
"""

import numpy as np
import torch


def eval_policy(actor, env):
    """
    Evaluate one actor for a full episode.
    
    Simplified from parallel_worker() in original PGA-MAP-Elites.
    Source: https://github.com/ollenilsson19/PGA-MAP-Elites
    
    Original (lines 68-70):
        state = env.reset()
        done = False
        while not done:
            action = actor.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
    
    Args:
        actor: Policy network with select_action() method
        env: Environment with reset(), step(), tot_reward, desc attributes
    
    Returns:
        fitness: Total episode reward (env.tot_reward)
        desc: Behavior descriptor (env.desc)
        transitions: Tuple of arrays (states, actions, next_states, rewards, not_dones)
                    Format matches ReplayBuffer.add() expected input
    """
    # Original line 68: state = env.reset()
    state = env.reset()
    done = False
    
    # Storage for transitions
    # Original lines 77-81 use vstack to accumulate arrays
    states = []
    actions = []
    next_states = []
    rewards = []
    not_dones = []
    
    # Original line 69-70: evaluation loop
    while not done:
        # Original line 71: action = actor.select_action(np.array(state))
        # select_action() handles torch conversion internally
        action = actor.select_action(np.array(state))
        
        # Original line 72: next_state, reward, done, _ = env.step(action)
        next_state, reward, done, info = env.step(action)
        
        # Original line 73: done_bool = float(done) if env.T < env._max_episode_steps else 0
        # This distinguishes true terminal states from timeout truncation
        # If episode ended due to time limit, don't treat as terminal for Q-learning
        # Source: TD3 paper - bootstrapping should continue at timeout
        if hasattr(env, 'T') and hasattr(env, '_max_episode_steps'):
            done_bool = float(done) if env.T < env._max_episode_steps else 0
        else:
            # Fallback: use done directly
            done_bool = float(done)
        
        # Original lines 74-81: accumulate transitions
        # Original uses vstack, we use lists then convert (simpler)
        states.append(state.copy())
        actions.append(action.copy())
        next_states.append(next_state.copy())
        rewards.append(reward)
        not_dones.append(1.0 - done_bool)  # ReplayBuffer expects not_done, not done
        
        # Original line 82: state = next_state
        state = next_state
    
    # Convert to arrays matching ReplayBuffer.add() format
    # Original line 85: trans_out_queue.put((idx, (state_array, action_array, ...)))
    # ReplayBuffer.add() expects tuple of arrays: (states, actions, next_states, rewards, dones)
    transitions = (
        np.array(states),
        np.array(actions),
        np.array(next_states),
        np.array(rewards).reshape(-1, 1),    # ReplayBuffer expects shape (n, 1)
        np.array(not_dones).reshape(-1, 1),  # ReplayBuffer expects shape (n, 1)
    )
    
    # Original line 83: eval_out_queue.put((idx, (env.tot_reward, env.desc, env.alive, env.T)))
    # Return fitness and descriptor for archive placement
    fitness = env.tot_reward
    desc = env.desc.copy()
    
    return fitness, desc, transitions


def eval_batch(actors, env):
    """
    Evaluate multiple actors sequentially.
    
    Simplified from ParallelEnv.eval_policy() which uses parallel workers.
    Original (lines 143-150):
        for idx, actor in enumerate(actors):
            self.eval_in_queue.put((idx, actor, ...))
        for _ in range(len(actors)):
            idx, result = self.eval_out_queue.get()
            results[idx] = result
    
    Args:
        actors: List of policy networks
        env: Environment instance (reused for all evaluations)
    
    Returns:
        results: List of (fitness, desc, transitions) for each actor
    """
    results = []
    for actor in actors:
        fitness, desc, transitions = eval_policy(actor, env)
        results.append((fitness, desc, transitions))
    return results