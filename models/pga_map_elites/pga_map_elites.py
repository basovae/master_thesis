"""""
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

[LOGGING ADDED] - Search for this tag to find all logging additions
"""""

import numpy as np
import torch
from sklearn.neighbors import KDTree
import os
import time  # [LOGGING ADDED] For timing

# Official utils.py (copy-pasted from original repo)
from .official_logic import (
    ReplayBuffer,
    Individual,
    add_to_archive,
    cvt,
    make_hashable,
    save_archive
)

# Your network implementations (keep your existing ones or use original)
from .networks import Actor, Critic


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
    "n_niches": 256,         # Number of archive cells (paper uses 1296)
    "max_evals": int(1e5),    # Total evaluations
    "cvt_samples": 25000,     # Samples for CVT computation
    "random_init": 500,       # Random evaluations before PG variation
    "eval_batch_size": 100,   # Batch size per iteration
    "save_period": 1000,      # Save archive every N evals
    
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
    
    # [LOGGING ADDED] Logging options
    "verbose": True,          # Enable detailed logging
    "log_interval": 1,        # Log every N iterations
}


# =============================================================================
# [LOGGING ADDED] Logging helper functions
# =============================================================================
def log_archive_metrics(archive, n_niches, bd_names=["volatility", "diversification"]):
    """Log detailed archive quality metrics."""
    if len(archive) == 0:
        print("  Archive: Empty")
        return {}
    
    fit_list = np.array([x.fitness for x in archive.values()])
    bd_list = np.array([x.desc for x in archive.values()])
    
    metrics = {
        'coverage': len(archive),
        'coverage_pct': 100 * len(archive) / n_niches,
        'max_fitness': fit_list.max(),
        'mean_fitness': np.mean(fit_list),
        'median_fitness': np.median(fit_list),
        'min_fitness': fit_list.min(),
        'std_fitness': np.std(fit_list),
        'qd_score': np.sum(fit_list),  # Sum of all fitnesses
    }
    
    # BD statistics
    for i, name in enumerate(bd_names):
        if bd_list.shape[1] > i:
            metrics[f'bd_{name}_mean'] = np.mean(bd_list[:, i])
            metrics[f'bd_{name}_std'] = np.std(bd_list[:, i])
            metrics[f'bd_{name}_min'] = np.min(bd_list[:, i])
            metrics[f'bd_{name}_max'] = np.max(bd_list[:, i])
    
    print(f"  Archive Metrics:")
    print(f"    Coverage: {metrics['coverage']}/{n_niches} ({metrics['coverage_pct']:.1f}%)")
    print(f"    Fitness: max={metrics['max_fitness']:.4f}, mean={metrics['mean_fitness']:.4f}, "
          f"median={metrics['median_fitness']:.4f}, std={metrics['std_fitness']:.4f}")
    print(f"    QD-Score: {metrics['qd_score']:.4f}")
    
    # BD distribution
    print(f"    BD Distribution:")
    for i, name in enumerate(bd_names):
        if bd_list.shape[1] > i:
            print(f"      {name}: mean={metrics[f'bd_{name}_mean']:.3f}, "
                  f"std={metrics[f'bd_{name}_std']:.3f}, "
                  f"range=[{metrics[f'bd_{name}_min']:.3f}, {metrics[f'bd_{name}_max']:.3f}]")
    
    return metrics


def log_replay_buffer(replay_buffer):
    """Log replay buffer status."""
    fill_pct = 100 * replay_buffer.size / replay_buffer.max_size
    print(f"  Replay Buffer: {replay_buffer.size:,}/{replay_buffer.max_size:,} ({fill_pct:.1f}%)")


def log_variation_stats(pg_added, pg_total, ga_added, ga_total):
    """Log variation operator success rates."""
    pg_rate = 100 * pg_added / pg_total if pg_total > 0 else 0
    ga_rate = 100 * ga_added / ga_total if ga_total > 0 else 0
    total_added = pg_added + ga_added
    total = pg_total + ga_total
    total_rate = 100 * total_added / total if total > 0 else 0
    
    print(f"  Variation Success:")
    print(f"    PG: {pg_added}/{pg_total} ({pg_rate:.1f}%)")
    print(f"    GA: {ga_added}/{ga_total} ({ga_rate:.1f}%)")
    print(f"    Total: {total_added}/{total} ({total_rate:.1f}%)")
    
    return {'pg_added': pg_added, 'pg_total': pg_total, 'pg_rate': pg_rate,
            'ga_added': ga_added, 'ga_total': ga_total, 'ga_rate': ga_rate}


# =============================================================================
# [ADDED] Simple evaluation function (replaces ParallelEnv)
# =============================================================================
def evaluate_policy(actor, env, max_steps=1000):
    """
    Evaluate a policy and collect transitions.
    
    Returns:
        fitness: Cumulative episode reward
        behavior_descriptor: BD in [0,1] range
        transitions: Tuple of arrays for replay buffer
        alive: Whether agent survived full episode
    """
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
    
    # [LOGGING ADDED] Track PG variation metrics
    initial_q = None
    final_q = None
    
    # Apply n_grad gradient steps
    for step in range(config["nr_of_steps_act"]):
        # Sample states from replay buffer
        states, _, _, _, _ = replay_buffer.sample(config["train_batch_size"])
        
        # Compute policy gradient: maximize Q(s, π(s))
        actions = offspring(states)
        q_values = critic.critic.Q1(states, actions)  # Use Q1 only for actor update
        
        # [LOGGING ADDED] Track Q-value improvement
        if step == 0:
            initial_q = q_values.mean().item()
        if step == config["nr_of_steps_act"] - 1:
            final_q = q_values.mean().item()
        
        actor_loss = -q_values.mean()
        
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
    
    # [ADDED] Track lineage
    offspring.parent_1_id = getattr(parent, 'id', -1)
    offspring.parent_2_id = None
    offspring.type = "pg"
    
    # [LOGGING ADDED] Store Q improvement on offspring
    offspring.pg_q_improvement = final_q - initial_q if (initial_q and final_q) else None
    
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
    
    # [LOGGING ADDED] Create log files
    log_file = open(f"{config['save_path']}/progress.dat", 'w')
    
    # [LOGGING ADDED] CSV file for detailed metrics
    metrics_file = open(f"{config['save_path']}/metrics.csv", 'w')
    metrics_file.write("evals,iteration,coverage,coverage_pct,max_fitness,mean_fitness,qd_score,"
                       "pg_added,pg_total,ga_added,ga_total,critic_loss,avg_q,buffer_size,"
                       "iter_time,phase\n")
    
    # [LOGGING ADDED] Track global metrics
    iteration = 0
    total_time = 0
    best_fitness_ever = float('-inf')
    
    # ==========================================================================
    # Initialize components (same as original)
    # ==========================================================================
    
    print("=" * 60)
    print("PGA-MAP-Elites Initialization")
    print("=" * 60)
    
    # Compute CVT centroids
    print(f"Computing CVT with {config['n_niches']} niches, {config['dim_map']}D behavior space...")
    cvt_start = time.time()
    centroids = cvt(
        config["n_niches"],
        config["dim_map"],
        config["cvt_samples"],
        cvt_use_cache=True
    )
    kdt = KDTree(centroids, leaf_size=30, metric='euclidean')
    print(f"  CVT computed in {time.time() - cvt_start:.1f}s")
    
    # Initialize archive (dict-based, like original)
    archive = {}
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config["state_dim"], config["action_dim"])
    print(f"  Replay buffer initialized (max size: {replay_buffer.max_size:,})")
    
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
    print(f"  Critic initialized (TD3-style twin critics)")
    
    print(f"\nConfig Summary:")
    print(f"  Max evaluations: {config['max_evals']:,}")
    print(f"  Random init: {config['random_init']} evals")
    print(f"  Batch size: {config['eval_batch_size']}")
    print(f"  Variation split: {config['proportion_evo']*100:.0f}% GA, {(1-config['proportion_evo'])*100:.0f}% PG")
    print(f"  Critic training: {config['nr_of_steps_crit']} steps/iter")
    print(f"  PG variation: {config['nr_of_steps_act']} steps/offspring")
    print("=" * 60)
    
    n_evals = 0
    b_evals = 0
    
    # ==========================================================================
    # Main MAP-Elites loop (same structure as original)
    # ==========================================================================
    
    while n_evals < config["max_evals"]:
        iteration += 1
        iter_start = time.time()  # [LOGGING ADDED]
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} | Evals: {n_evals}/{config['max_evals']}")
        print(f"{'='*60}")
        
        to_evaluate = []
        phase = "random" if n_evals < config["random_init"] else "variation"
        
        # [LOGGING ADDED] Track variation success
        pg_offspring = []
        ga_offspring = []
        
        # ======================================================================
        # Random initialization phase
        # ======================================================================
        if n_evals < config["random_init"]:
            print(f"Phase: Random Initialization ({n_evals}/{config['random_init']})")
            for _ in range(config["eval_batch_size"]):
                # Create random actor
                actor = Actor(
                    config["state_dim"],
                    config["action_dim"],
                    config["max_action"],
                    config["neurons_list"]
                )
                actor.type = "random"
                to_evaluate.append(actor)
        
        # ======================================================================
        # Selection and variation phase
        # ======================================================================
        else:
            print(f"Phase: Selection & Variation")
            
            # Train critic (Algorithm 4 from paper)
            critic_metrics = {}
            if replay_buffer.size > config["train_batch_size"]:
                print(f"\n  [Critic Training]")
                critic_start = time.time()
                
                # [LOGGING ADDED] Use verbose mode for critic
                critic_metrics = critic.train(
                    archive, 
                    replay_buffer, 
                    config["nr_of_steps_crit"], 
                    config["train_batch_size"],
                    verbose=config.get("verbose", False)
                )
                
                critic_time = time.time() - critic_start
                print(f"    Time: {critic_time:.1f}s")
                if critic_metrics:
                    print(f"    Final Critic Loss: {critic_metrics.get('critic_loss', 0):.4f}")
                    print(f"    Avg Q-value: {critic_metrics.get('avg_q', 0):.4f}")
            
            # Determine split between GA and PG variation
            n_evo = int(config["eval_batch_size"] * config["proportion_evo"])
            n_pg = config["eval_batch_size"] - n_evo
            
            # Sample parents from archive
            archive_keys = list(archive.keys())
            
            # --- PG Variation ---
            if n_pg > 0 and replay_buffer.size > config["train_batch_size"]:
                print(f"\n  [PG Variation] Generating {n_pg} offspring...")
                pg_start = time.time()
                
                for i in range(n_pg):
                    parent_key = archive_keys[np.random.randint(len(archive_keys))]
                    parent = archive[parent_key].x
                    offspring = pg_variation(parent, critic, replay_buffer, config)
                    to_evaluate.append(offspring)
                    pg_offspring.append(offspring)
                    
                    # [LOGGING ADDED] Progress for long PG generation
                    if (i + 1) % 10 == 0:
                        print(f"    Generated {i+1}/{n_pg} PG offspring...")
                
                print(f"    Time: {time.time() - pg_start:.1f}s")
            
            # --- GA Variation (iso_dd) ---
            print(f"\n  [GA Variation] Generating {n_evo} offspring...")
            ga_start = time.time()
            
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
                ga_offspring.append(offspring)
            
            print(f"    Time: {time.time() - ga_start:.1f}s")
        
        # ======================================================================
        # Evaluate batch and add to archive
        # ======================================================================
        print(f"\n  [Evaluation] Evaluating {len(to_evaluate)} policies...")
        eval_start = time.time()
        
        # [LOGGING ADDED] Track additions by type
        pg_added = 0
        ga_added = 0
        random_added = 0
        
        # [LOGGING ADDED] Track fitness/BD of evaluated policies
        eval_fitnesses = []
        eval_bds = []
        
        for idx, actor in enumerate(to_evaluate):
            fitness, bd, transitions, alive = evaluate_policy(actor, env)
            
            # Add transitions to replay buffer
            replay_buffer.add(transitions)
            
            # [ADDED] Normalize BD to [0,1] for CVT lookup
            bd_normalized = normalize_bd(bd, config["bd_bounds"])
            
            # Create Individual and try to add to archive
            individual = Individual(actor, bd_normalized, fitness)
            was_added = add_to_archive(individual, bd_normalized, archive, kdt)
            
            # [LOGGING ADDED] Track by type
            if was_added:
                if actor.type == "pg":
                    pg_added += 1
                elif actor.type == "iso_dd":
                    ga_added += 1
                elif actor.type == "random":
                    random_added += 1
            
            # [LOGGING ADDED] Collect evaluation stats
            eval_fitnesses.append(fitness)
            eval_bds.append(bd_normalized)
            
            # [LOGGING ADDED] Progress for long evaluations
            if (idx + 1) % 25 == 0:
                print(f"    Evaluated {idx+1}/{len(to_evaluate)}...")
        
        eval_time = time.time() - eval_start
        print(f"    Time: {eval_time:.1f}s")
        
        # [LOGGING ADDED] Evaluation batch statistics
        eval_fitnesses = np.array(eval_fitnesses)
        eval_bds = np.array(eval_bds)
        print(f"    Batch fitness: mean={np.mean(eval_fitnesses):.4f}, "
              f"max={np.max(eval_fitnesses):.4f}, min={np.min(eval_fitnesses):.4f}")
        
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)
        
        # ======================================================================
        # Logging and saving
        # ======================================================================
        iter_time = time.time() - iter_start
        total_time += iter_time
        
        print(f"\n  [Summary]")
        print(f"    Iteration time: {iter_time:.1f}s (total: {total_time:.1f}s)")
        
        # [LOGGING ADDED] Log archive metrics
        archive_metrics = log_archive_metrics(archive, config["n_niches"])
        
        # [LOGGING ADDED] Log replay buffer
        log_replay_buffer(replay_buffer)
        
        # [LOGGING ADDED] Log variation success (only in variation phase)
        if phase == "variation":
            var_stats = log_variation_stats(
                pg_added, len(pg_offspring),
                ga_added, len(ga_offspring)
            )
        else:
            print(f"  Random policies added: {random_added}/{len(to_evaluate)}")
        
        # [LOGGING ADDED] Track best fitness
        if archive_metrics and archive_metrics.get('max_fitness', 0) > best_fitness_ever:
            best_fitness_ever = archive_metrics['max_fitness']
            print(f"  *** New best fitness: {best_fitness_ever:.4f} ***")
        
        # Write to log files
        if len(archive) > 0:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write(f"{n_evals} {len(archive)} {fit_list.max():.4f} "
                          f"{np.sum(fit_list):.4f} {np.mean(fit_list):.4f}\n")
            log_file.flush()
            
            # [LOGGING ADDED] Write detailed metrics CSV
            critic_loss = critic_metrics.get('critic_loss', 0) if 'critic_metrics' in dir() else 0
            avg_q = critic_metrics.get('avg_q', 0) if 'critic_metrics' in dir() else 0
            pg_add = pg_added if phase == "variation" else 0
            pg_tot = len(pg_offspring) if phase == "variation" else 0
            ga_add = ga_added if phase == "variation" else random_added
            ga_tot = len(ga_offspring) if phase == "variation" else len(to_evaluate)
            
            metrics_file.write(f"{n_evals},{iteration},{len(archive)},{archive_metrics['coverage_pct']:.2f},"
                              f"{archive_metrics['max_fitness']:.4f},{archive_metrics['mean_fitness']:.4f},"
                              f"{archive_metrics['qd_score']:.4f},{pg_add},{pg_tot},{ga_add},{ga_tot},"
                              f"{critic_loss:.4f},{avg_q:.4f},{replay_buffer.size},"
                              f"{iter_time:.2f},{phase}\n")
            metrics_file.flush()
        
        # Save archive periodically
        if b_evals >= config["save_period"]:
            print(f"\n  [Saving] Archive checkpoint at {n_evals} evals...")
            save_archive(archive, n_evals, "pga_me", config["save_path"])
            b_evals = 0
    
    # ==========================================================================
    # Final summary
    # ==========================================================================
    print(f"\n{'='*60}")
    print("PGA-MAP-Elites Complete!")
    print(f"{'='*60}")
    print(f"Total evaluations: {n_evals:,}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg time per iteration: {total_time/iteration:.1f}s")
    
    log_archive_metrics(archive, config["n_niches"])
    
    # Final save
    save_archive(archive, n_evals, "pga_me", config["save_path"], save_models=True)
    log_file.close()
    metrics_file.close()
    
    # [LOGGING ADDED] Save critic training history
    if hasattr(critic, 'get_training_history'):
        history = critic.get_training_history()
        np.savez(f"{config['save_path']}/critic_history.npz", **history)
        print(f"  Critic training history saved to {config['save_path']}/critic_history.npz")
    
    return archive


# =============================================================================
# [ADDED] Example usage with dummy environment
# =============================================================================
if __name__ == "__main__":
    import numpy as np
    
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
    
    # [LOGGING ADDED] Enable verbose logging
    config["verbose"] = True
    
    # Run algorithm
    print("Starting PGA-MAP-Elites...")
    archive = main(env, config)
    
    print(f"\nFinal Results:")
    print(f"  Archive size: {len(archive)}")
    if len(archive) > 0:
        fit_list = np.array([x.fitness for x in archive.values()])
        print(f"  Max fitness: {fit_list.max():.4f}")
        print(f"  Coverage: {100*len(archive)/config['n_niches']:.1f}%")