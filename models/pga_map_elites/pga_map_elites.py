"""
PGA-MAP-Elites for Portfolio Optimization
Adapted from official implementation: https://github.com/ollenilsson19/PGA-MAP-Elites

CHANGES FROM ORIGINAL:
1. Removed QDgym dependency - uses generic gym-style env
2. Removed parallel environments - single env for simplicity (can add back later)
3. Removed argparse - uses config dict instead
4. Removed species archive - not used in GECCO paper (n_species=1)
5. Simplified actor/critic creation - inline instead of partials
6. Added behavior descriptor normalization to [0,1] for CVT
7. Removed vectorized_env dependency - sequential evaluation
8. Added comments marking all changes with [CHANGED] or [REMOVED] or [ADDED]
"""

import numpy as np
import torch
from sklearn.neighbors import KDTree
import os

# Official utils.py (copy-pasted from original repo)
from official_logic import (
    ReplayBuffer,
    Individual,
    add_to_archive,
    cvt,
    make_hashable,
    save_archive
)

# Your network implementations (keep your existing ones or use original)
from networks import Actor, Critic


# =============================================================================
# [CHANGED] Config dict instead of argparse
# =============================================================================
config = {
    # Environment
    "state_dim": 10,          # Set to match your portfolio env
    "action_dim": 5,          # Number of assets
    "max_action": 1.0,        # Portfolio weights typically [0,1]
    
    # QD params
    "dim_map": 2,             # Behavior descriptor dimension (e.g., volatility, diversification)
    "n_niches": 1024,         # Number of archive cells (paper uses 1296)
    "max_evals": int(1e5),    # Total evaluations
    "cvt_samples": 25000,     # Samples for CVT computation
    "random_init": 500,       # Random evaluations before PG variation
    "eval_batch_size": 100,   # Batch size per iteration
    "save_period": 10000,     # Save archive every N evals
    
    # [ADDED] Behavior descriptor bounds for normalization
    "bd_bounds": ([0.0, 0.0], [0.3, 1.0]),  # (min, max) for each BD dimension
    
    # GA params (iso_dd / directional variation)
    "iso_sigma": 0.01,        # Gaussian perturbation (sigma_1)
    "line_sigma": 0.2,        # Directional component (sigma_2)
    "proportion_evo": 0.5,    # Proportion using GA vs PG variation
    
    # RL params (TD3-style)
    "train_batch_size": 256,  # Batch size for critic training
    "discount": 0.99,         # Gamma
    "tau": 0.005,             # Target network update rate
    "policy_noise": 0.2,      # Target policy smoothing noise
    "noise_clip": 0.5,        # Noise clipping range
    "policy_freq": 2,         # Delayed policy updates
    "nr_of_steps_crit": 300,  # Critic training steps per iteration (n_crit)
    "nr_of_steps_act": 10,    # PG variation steps (n_grad)
    "lr": 0.001,              # Learning rate for PG variation
    
    # Network architecture
    "neurons_list": [128, 128],  # Hidden layers for actor
    
    # Paths
    "save_path": "./results",
    "seed": 0,
}


# =============================================================================
# [ADDED] Simple evaluation function (replaces ParallelEnv)
# =============================================================================
def evaluate_policy(actor, env, max_steps=1000):
    states, actions, next_states, rewards, dones = [], [], [], [], []
    
    state = env.reset()
    episode_reward = 0
    portfolio_returns = []
    
    for step in range(max_steps):
        with torch.no_grad():
            action = actor(torch.FloatTensor(state).unsqueeze(0)).cpu().numpy().flatten()
        
        next_state, reward, done, info = env.step(action)
        
        # Collect transitions
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(float(done))
        
        # Track for BD computation
        portfolio_returns.append(info.get('portfolio_return', reward))
        
        episode_reward += reward
        state = next_state
        if done:
            break
    
    # === Compute BD here ===
    weights = np.array(actions)
    returns = np.array(portfolio_returns)
    
    # Volatility in [0,1]
    volatility = np.clip(np.std(returns) / 0.3, 0, 1)
    
    # Diversification in [0,1]
    avg_weights = np.mean(weights, axis=0)
    diversification = 1.0 - np.sum(avg_weights ** 2)
    
    behavior_descriptor = np.array([volatility, diversification])
    
    # Pack transitions
    transitions = (
        np.array(states),
        np.array(actions),
        np.array(next_states),
        np.array(rewards).reshape(-1, 1),
        np.array(dones).reshape(-1, 1)
    )
    
    return episode_reward, behavior_descriptor, transitions, (step == max_steps - 1)


# =============================================================================
# [ADDED] Normalize behavior descriptor to [0,1] for CVT lookup
# =============================================================================
def normalize_bd(bd, bd_bounds):
    """Normalize BD to [0,1] range for CVT archive."""
    bd_min = np.array(bd_bounds[0])
    bd_max = np.array(bd_bounds[1])
    return np.clip((bd - bd_min) / (bd_max - bd_min + 1e-8), 0, 1)


# =============================================================================
# [CHANGED] Simplified iso_dd variation (extracted from VariationalOperator)
# =============================================================================
def iso_dd_variation(parent1, parent2, iso_sigma=0.01, line_sigma=0.2):
    """
    Directional variation operator (Vassiliades & Mouret, 2018).
    offspring = parent1 + N(0, iso_sigma) + N(0, line_sigma) * (parent2 - parent1)
    """
    # Create offspring as copy of parent1
    offspring = type(parent1)(
        parent1.state_dim,
        parent1.action_dim,
        parent1.max_action,
        parent1.neurons_list
    )
    offspring.load_state_dict(parent1.state_dict())
    
    # Apply variation to each parameter
    with torch.no_grad():
        for p1, p2, p_off in zip(parent1.parameters(), 
                                   parent2.parameters(), 
                                   offspring.parameters()):
            iso_noise = torch.randn_like(p1) * iso_sigma
            line_noise = torch.randn(1).item() * line_sigma
            p_off.data = p1.data + iso_noise + line_noise * (p2.data - p1.data)
    
    # [ADDED] Track lineage for debugging
    offspring.parent_1_id = getattr(parent1, 'id', -1)
    offspring.parent_2_id = getattr(parent2, 'id', -1)
    offspring.type = "iso_dd"
    
    return offspring


# =============================================================================
# [CHANGED] Simplified PG variation (extracted from VariationalOperator)
# =============================================================================
def pg_variation(parent, critic, replay_buffer, config):
    """
    Policy gradient variation using TD3-style critic.
    Applies n_grad steps of gradient ascent on Q-values.
    """
    # Create offspring as copy of parent
    offspring = type(parent)(
        parent.state_dim,
        parent.action_dim,
        parent.max_action,
        parent.neurons_list
    )
    offspring.load_state_dict(parent.state_dict())
    
    # Optimizer for PG updates
    optimizer = torch.optim.Adam(offspring.parameters(), lr=config["lr"])
    
    # Apply n_grad gradient steps
    for _ in range(config["nr_of_steps_act"]):
        # Sample states from replay buffer
        states, _, _, _, _ = replay_buffer.sample(config["train_batch_size"])
        
        # Compute policy gradient: maximize Q(s, π(s))
        actions = offspring(states)
        q_values = critic.Q1(states, actions)  # Use Q1 only for actor update
        
        actor_loss = -q_values.mean()
        
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
    
    # [ADDED] Track lineage
    offspring.parent_1_id = getattr(parent, 'id', -1)
    offspring.parent_2_id = None
    offspring.type = "pg"
    
    return offspring


# =============================================================================
# [CHANGED] Main loop - simplified from original
# =============================================================================
def main(env, config):
    """
    Main PGA-MAP-Elites loop.
    
    Args:
        env: Gym-style environment
        config: Configuration dictionary
    """
    # Set seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Create save directory
    if not os.path.exists(config["save_path"]):
        os.makedirs(f"{config['save_path']}/models/", exist_ok=True)
    
    # Log file
    log_file = open(f"{config['save_path']}/progress.dat", 'w')
    
    # ==========================================================================
    # Initialize components (same as original)
    # ==========================================================================
    
    # Compute CVT centroids
    centroids = cvt(
        config["n_niches"],
        config["dim_map"],
        config["cvt_samples"],
        cvt_use_cache=True
    )
    kdt = KDTree(centroids, leaf_size=30, metric='euclidean')
    
    # Initialize archive (dict-based, like original)
    archive = {}
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config["state_dim"], config["action_dim"])
    
    # Initialize critic (TD3-style twin critics)
    critic = Critic(
        config["state_dim"],
        config["action_dim"],
        config["max_action"],
        discount=config["discount"],
        tau=config["tau"],
        policy_noise=config["policy_noise"],
        noise_clip=config["noise_clip"],
        policy_freq=config["policy_freq"]
    )
    
    # [REMOVED] Greedy controller - using archive policies directly instead
    
    n_evals = 0
    b_evals = 0
    
    # ==========================================================================
    # Main MAP-Elites loop (same structure as original)
    # ==========================================================================
    
    while n_evals < config["max_evals"]:
        print(f"Archive size: {len(archive)}")
        to_evaluate = []
        
        # ======================================================================
        # Random initialization phase
        # ======================================================================
        if n_evals < config["random_init"]:
            print("Random initialization phase")
            for _ in range(config["eval_batch_size"]):
                # Create random actor
                actor = Actor(
                    config["state_dim"],
                    config["action_dim"],
                    config["max_action"],
                    config["neurons_list"]
                )
                to_evaluate.append(actor)
        
        # ======================================================================
        # Selection and variation phase
        # ======================================================================
        else:
            print("Selection/Variation phase")
            
            # Train critic (Algorithm 4 from paper)
            if replay_buffer.size > config["train_batch_size"]:
                print(f"  Training critic for {config['nr_of_steps_crit']} steps...")
                for _ in range(config["nr_of_steps_crit"]):
                    critic.train_step(replay_buffer, config["train_batch_size"])
            
            # Determine split between GA and PG variation
            n_evo = int(config["eval_batch_size"] * config["proportion_evo"])
            n_pg = config["eval_batch_size"] - n_evo
            
            # Sample parents from archive
            archive_keys = list(archive.keys())
            
            # --- PG Variation ---
            if n_pg > 0 and replay_buffer.size > config["train_batch_size"]:
                print(f"  Generating {n_pg} PG offspring...")
                for _ in range(n_pg):
                    parent_key = archive_keys[np.random.randint(len(archive_keys))]
                    parent = archive[parent_key].x
                    offspring = pg_variation(parent, critic, replay_buffer, config)
                    to_evaluate.append(offspring)
            
            # --- GA Variation (iso_dd) ---
            print(f"  Generating {n_evo} GA offspring...")
            for _ in range(n_evo):
                # Select two parents
                idx1, idx2 = np.random.randint(len(archive_keys), size=2)
                parent1 = archive[archive_keys[idx1]].x
                parent2 = archive[archive_keys[idx2]].x
                offspring = iso_dd_variation(
                    parent1, parent2,
                    config["iso_sigma"],
                    config["line_sigma"]
                )
                to_evaluate.append(offspring)
        
        # ======================================================================
        # Evaluate batch and add to archive
        # ======================================================================
        print(f"  Evaluating {len(to_evaluate)} policies...")
        for actor in to_evaluate:
            fitness, bd, transitions, alive = evaluate_policy(actor, env)
            
            # Add transitions to replay buffer
            replay_buffer.add(transitions)
            
            # [ADDED] Normalize BD to [0,1] for CVT lookup
            bd_normalized = normalize_bd(bd, config["bd_bounds"])
            
            # Create Individual and try to add to archive
            individual = Individual(actor, bd_normalized, fitness)
            add_to_archive(individual, bd_normalized, archive, kdt)
        
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)
        print(f"[{n_evals}/{config['max_evals']}]")
        
        # ======================================================================
        # Logging and saving
        # ======================================================================
        if len(archive) > 0:
            fit_list = np.array([x.fitness for x in archive.values()])
            print(f"  Max fitness: {fit_list.max():.4f}")
            print(f"  Mean fitness: {np.mean(fit_list):.4f}")
            print(f"  Coverage: {len(archive)}/{config['n_niches']} ({100*len(archive)/config['n_niches']:.1f}%)")
            
            log_file.write(f"{n_evals} {len(archive)} {fit_list.max():.4f} "
                          f"{np.sum(fit_list):.4f} {np.mean(fit_list):.4f}\n")
            log_file.flush()
        
        # Save archive periodically
        if b_evals >= config["save_period"]:
            save_archive(archive, n_evals, "pga_me", config["save_path"])
            b_evals = 0
    
    # Final save
    save_archive(archive, n_evals, "pga_me", config["save_path"], save_models=True)
    log_file.close()
    
    return archive


# =============================================================================
# [ADDED] Example usage with dummy environment
# =============================================================================
if __name__ == "__main__":
    
    # Dummy environment for testing (replace with your portfolio env)
    class DummyPortfolioEnv:
        def __init__(self):
            self.state_dim = 10
            self.action_dim = 5
            self.max_steps = 50
            self.step_count = 0
        
        def reset(self):
            self.step_count = 0
            return np.random.randn(self.state_dim)
        
        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(self.state_dim)
            reward = np.random.randn() * 0.01
            done = self.step_count >= self.max_steps
            info = {'portfolio_return': reward}
            return next_state, reward, done, info
    
    env = DummyPortfolioEnv()
    
    # Update config to match env
    config["state_dim"] = env.state_dim
    config["action_dim"] = env.action_dim
    
    # Run algorithm
    print("Starting PGA-MAP-Elites...")
    archive = main(env, config)
    
    print(f"\nFinal Results:")
    print(f"  Archive size: {len(archive)}")
    if len(archive) > 0:
        fit_list = np.array([x.fitness for x in archive.values()])
        print(f"  Max fitness: {fit_list.max():.4f}")
        print(f"  Coverage: {100*len(archive)/config['n_niches']:.1f}%")