# =============================================================================
# variational_operators.py (simplified from Nilsson repo)
# =============================================================================
# Original: https://github.com/ollenilsson19/PGA-MAP-Elites
# Removed: multiprocessing, CloudpickleWrapper dependency
# Kept: iso_dd crossover, mutation operators, same interface
# =============================================================================

import copy
import numpy as np
import torch


class VariationalOperator:
    """
    Variation operators for PGA-MAP-Elites.
    
    Two types:
    - GA variation: iso_dd crossover (Vassiliades & Mouret, 2018)
    - PG variation: gradient ascent on Q-value
    
    Args:
        actor_fn: Function that returns a new Actor instance
        iso_sigma: Isotropic noise std (default 0.005)
        line_sigma: Line noise std (default 0.05)
        learning_rate: LR for PG variation (default 3e-4)
    """
    def __init__(
        self,
        actor_fn,
        iso_sigma=0.005,
        line_sigma=0.05,
        learning_rate=3e-4,
    ):
        self.actor_fn = actor_fn
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self.learning_rate = learning_rate

    def __call__(
        self,
        archive,
        batch_size,
        proportion_evo,
        critic=None,
        states=None,
        nr_of_steps_act=10,
    ):
        """
        Generate offspring from archive.
        
        Args:
            archive: Dict of Individuals
            batch_size: Number of offspring to generate
            proportion_evo: Fraction using GA (rest uses PG)
            critic: Critic for PG variation
            states: List of state batches for PG variation
            nr_of_steps_act: Gradient steps for PG variation
            
        Returns:
            List of offspring actors
        """
        keys = list(archive.keys())
        offspring = []
        
        n_ga = int(batch_size * proportion_evo)
        n_pg = batch_size - n_ga
        
        # GA variation (iso_dd crossover)
        for _ in range(n_ga):
            idx1, idx2 = np.random.randint(len(keys), size=2)
            parent1 = archive[keys[idx1]].x
            parent2 = archive[keys[idx2]].x
            child = self._ga_variation(parent1, parent2)
            offspring.append(child)
        
        # PG variation (gradient ascent on Q)
        for _ in range(n_pg):
            idx = np.random.randint(len(keys))
            parent = archive[keys[idx]].x
            child = self._pg_variation(parent, critic, states, nr_of_steps_act)
            offspring.append(child)
        
        return offspring

    def _ga_variation(self, parent1, parent2):
        """
        Iso+Line crossover (Vassiliades & Mouret, 2018).
        child = parent1 + iso_noise + line_noise * (parent2 - parent1)
        """
        child = copy.deepcopy(parent1)
        child.type = "evo"
        child.parent_1_id = getattr(parent1, 'id', None)
        child.parent_2_id = getattr(parent2, 'id', None)
        
        child_state = child.state_dict()
        p1_state = parent1.state_dict()
        p2_state = parent2.state_dict()
        
        for key in child_state:
            if "weight" in key or "bias" in key:
                child_state[key] = self._iso_dd(p1_state[key], p2_state[key])
        
        child.load_state_dict(child_state)
        return child

    def _iso_dd(self, x, y):
        """
        Iso+Line operator on tensors.
        Ref: Vassiliades & Mouret, GECCO 2018
        """
        iso_noise = torch.zeros_like(x).normal_(mean=0, std=self.iso_sigma)
        line_noise = np.random.normal(0, self.line_sigma)
        return x.clone() + iso_noise + line_noise * (y - x)

    def _pg_variation(self, parent, critic, states, nr_of_steps):
        """
        Policy gradient variation.
        Apply nr_of_steps gradient ascent steps on Q-value.
        """
        child = copy.deepcopy(parent)
        child.type = "grad"
        child.parent_1_id = getattr(parent, 'id', None)
        child.parent_2_id = None
        
        # Enable gradients
        for param in child.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(child.parameters(), lr=self.learning_rate)
        
        # Gradient ascent on Q
        for i in range(nr_of_steps):
            state = states[i % len(states)]  # Cycle through states
            actor_loss = -critic.Q1(state, child(state)).mean()
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
        
        # Disable gradients for storage
        for param in child.parameters():
            param.requires_grad = False
        
        return child

    def close(self):
        """Compatibility with original (no-op without multiprocessing)."""
        pass