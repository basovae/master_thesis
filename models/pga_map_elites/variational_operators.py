import copy

import torch

def ga_variation(parent1, parent2, iso_sigma=0.005, line_sigma=0.05):
    """
    Iso-line (directional) variation operator.
    
    📝 PAPER (Algorithm 5, lines 10-12, Equation 6):
       φ̂ = φ1 + σ1·N(0,I) + σ2·(φ2-φ1)·N(0,1)
    
    📝 OFFICIAL CODE: variational_operators.py
       iso_sigma = 0.005 (isotropic noise)
       line_sigma = 0.05 (directional displacement)
    
    🎓 SIMPLIFICATION: None - this is already simple!
    """
    params1 = parent1.get_flat_params()
    params2 = parent2.get_flat_params()
    
    # Isotropic Gaussian noise
    iso_noise = torch.randn_like(params1) * iso_sigma
    
    # Directional component toward parent2
    direction = params2 - params1
    line_noise = torch.randn(1).item() * line_sigma
    
    # Offspring = parent1 + noise + direction
    offspring_params = params1 + iso_noise + line_noise * direction
    
    # Create new network with offspring params
    offspring = copy.deepcopy(parent1)
    offspring.set_flat_params(offspring_params)
    
    return offspring


def pg_variation(parent, critic, replay_buffer, n_grad=10, lr=0.001, batch_size=256):
    """
    Policy Gradient variation operator.
    
    📝 PAPER (Algorithm 5, lines 13-18):
       Apply n_grad steps of gradient descent maximizing Q1
       ∇_φ J(φ) = (1/N) Σ ∇_φ π_φ(s) ∇_a Q_θ1(s,a)|_{a=π_φ(s)}
    
    📝 OFFICIAL CODE: 
       n_grad = 10-50 (we use 10 for speed)
       lr = 0.001
       Uses Adam optimizer
    
    🎓 SIMPLIFICATION: 
       - Fixed 10 gradient steps (paper uses 10-50)
       - Could reduce batch_size for testing
    """
    offspring = copy.deepcopy(parent)
    optimizer = torch.optim.Adam(offspring.parameters(), lr=lr)
    
    for _ in range(n_grad):
        # Sample from replay buffer
        states, _, _, _, _ = replay_buffer.sample(batch_size)
        
        # Policy gradient: maximize Q1(s, π(s))
        actions = offspring(states)
        q_value = critic.Q1(states, actions)
        
        # Loss = -Q (we want to maximize Q)
        loss = -q_value.mean()
        
        optimizer.zero_grad()
        loss.backward()
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