import copy

import torch


def variation_ga(parent1, parent2, iso_sigma=0.005, line_sigma=0.05):
    """
    Algorithm 5, lines 10-12: VARIATION_GA
    Directional variation (Vassiliades & Mouret, 2018)
    
    φ̂ = φ1 + σ1·N(0, I) + σ2·(φ2 - φ1)·N(0, 1)
    
    Args:
        parent1, parent2: Parent policy networks
        iso_sigma: σ1 = 0.005 (isotropic noise)
        line_sigma: σ2 = 0.05 (directional displacement)
    
    Returns:
        offspring: New policy network
    """
    # Get flattened parameters
    params1 = parent1.get_params()
    params2 = parent2.get_params()
    
    # Isotropic Gaussian noise
    iso_noise = torch.randn_like(params1) * iso_sigma
    
    # Directional component (interpolation toward parent2)
    direction = params2 - params1
    line_noise = torch.randn(1).item() * line_sigma
    
    # Offspring parameters (Equation 6 in paper)
    offspring_params = params1 + iso_noise + line_noise * direction
    
    # Create offspring network
    offspring = copy.deepcopy(parent1)
    offspring.set_params(offspring_params)
    
    return offspring

def variation_pg(parent, critic, replay_buffer, n_grad=10, batch_size=256, lr=0.001):
    """
    Algorithm 5, lines 13-18: VARIATION_PG
    Apply n_grad steps of policy gradient
    
    Args:
        parent: Parent policy to mutate
        critic: Trained Q_θ1 network
        replay_buffer: Experience buffer B
        n_grad: Number of gradient steps (default 10, paper range 10-50)
        batch_size: N = 256
        lr: Learning rate for PG updates (0.001 in paper)
    
    Returns:
        offspring: Policy mutated via PG
    """
    offspring = copy.deepcopy(parent)
    optimizer = torch.optim.Adam(offspring.parameters(), lr=lr)
    
    for _ in range(n_grad):
        # Sample transitions (line 16)
        state, _, _, _, _ = replay_buffer.sample(batch_size)
        
        # Policy gradient: maximize Q_θ1(s, π(s)) (line 17)
        # ∇_φ J(φ) = (1/N) Σ ∇_φ π_φ(s) ∇_a Q_θ1(s, a)|_{a=π_φ(s)}
        actor_loss = -critic.Q1(state, offspring(state)).mean()
        
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
    
    return offspring

def variation(batch_size, archive, critic, replay_buffer, proportion_evo=0.5, n_grad=10):
    """
    Algorithm 5: VARIATION procedure
    
    Args:
        batch_size: Total number of offspring (b - 1, reserving 1 for greedy)
        archive: CVT-MAP-Elites archive X
        critic: Trained Q_θ1 network
        replay_buffer: Experience buffer B
        proportion_evo: Fraction using GA (default 0.5)
        n_grad: PG gradient steps
    
    Returns:
        offspring: List of mutated policies
    """
    n_evo = int(batch_size * proportion_evo)
    n_pg = batch_size - n_evo
    offspring = []
    
    # GA variation (first n_evo offspring)
    for _ in range(n_evo):
        parents = archive.uniform_selection(n=2)
        if len(parents) >= 2:
            child = variation_ga(parents[0], parents[1])
        else:
            child = variation_ga(parents[0], parents[0])
        offspring.append(child)
    
    # PG variation (remaining offspring)
    for _ in range(n_pg):
        parent = archive.uniform_selection(n=1)[0]
        child = variation_pg(parent, critic, replay_buffer, n_grad=n_grad)
        offspring.append(child)
    
    return offspring