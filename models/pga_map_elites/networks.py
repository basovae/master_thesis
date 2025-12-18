'''
Copyright (c) 2020 Scott Fujimoto
Based on Twin Delayed Deep Deterministic Policy Gradients (TD3)
Implementation by Scott Fujimoto https://github.com/sfujim/TD3 Paper: https://arxiv.org/abs/1802.09477
Modified by Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''
"""
PGA-MAP-Elites Networks with Logging
Based on official implementation: https://github.com/ollenilsson19/PGA-MAP-Elites

[LOGGING ADDED] - Search for this tag to find all logging additions
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    Policy network for continuous control.
    Architecture: [state_dim] -> [neurons_list] -> [action_dim]
    Output: tanh scaled by max_action
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        neurons_list,
        normalise=False,
        affine=True,
        weight_init_fn=nn.init.xavier_uniform_
    ):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.normalise = normalise
        self.affine = affine
        self.weight_init_fn = weight_init_fn
        self.num_layers = len(neurons_list)
        self.neurons_list = neurons_list
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Track lineage for debugging
        self.id = None
        self.parent_1_id = None
        self.parent_2_id = None
        self.type = "random"  # 'random', 'pg', 'iso_dd', 'critic_training'
        self.novel = None
        self.delta_f = None

        if self.num_layers == 1:
            self.l1 = nn.Linear(state_dim, neurons_list[0])
            self.l2 = nn.Linear(neurons_list[0], action_dim)
            if normalise:
                self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=affine)
                self.n2 = nn.LayerNorm(action_dim, elementwise_affine=affine)

        if self.num_layers == 2:
            self.l1 = nn.Linear(state_dim, neurons_list[0])
            self.l2 = nn.Linear(neurons_list[0], neurons_list[1])
            self.l3 = nn.Linear(neurons_list[1], action_dim)
            if normalise:
                self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=affine)
                self.n2 = nn.LayerNorm(neurons_list[1], elementwise_affine=affine)
                self.n3 = nn.LayerNorm(action_dim, elementwise_affine=affine)

        self.apply(self.init_weights)

    def forward(self, state):
        if self.num_layers == 1:
            if self.normalise:
                a = F.relu(self.n1(self.l1(state)))
                return self.max_action * torch.tanh(self.n2(self.l2(a)))
            else:
                a = F.relu(self.l1(state))
                return self.max_action * torch.tanh(self.l2(a))
        if self.num_layers == 2:
            if self.normalise:
                a = F.relu(self.n1(self.l1(state)))
                a = F.relu(self.n2(self.l2(a)))
                return self.max_action * torch.tanh(self.n3(self.l3(a)))
            else:
                a = F.relu(self.l1(state))
                a = F.relu(self.l2(a))
                return self.max_action * torch.tanh(self.l3(a))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self(state).cpu().data.numpy().flatten()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.weight_init_fn(m.weight)
        if isinstance(m, nn.LayerNorm):
            pass

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def return_copy(self):
        return copy.deepcopy(self)


class CriticNetwork(nn.Module):
    """
    Twin Q-network for TD3.
    Architecture: [state_dim + action_dim] -> [256, 256] -> [1]
    Returns: Q1, Q2 values
    """
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class Critic(object):
    """
    TD3-style critic training for PGA-MAP-Elites.
    
    Key features:
    - Twin critics (Q1, Q2) with clipped double Q-learning
    - Multiple actors from species archive trained together
    - Target networks with soft updates
    - Delayed policy updates (every d=2 steps)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        
        # Species actors for training
        self.actors_set = set()
        self.actors = []
        self.actor_targets = []
        self.actor_optimisers = []
        
        # [LOGGING ADDED] Track metrics over training
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []

    def train(self, archive, replay_buffer, nr_of_steps, batch_size=256, verbose=False):
        """
        Train critic and species actors for nr_of_steps iterations.
        
        Args:
            archive: Current MAP-Elites archive (dict)
            replay_buffer: Experience replay buffer
            nr_of_steps: Number of training iterations (n_crit in paper, default 300)
            batch_size: Batch size for sampling (N in paper, default 256)
            verbose: [LOGGING ADDED] Print detailed training info
        
        Returns:
            dict: Training metrics (critic_loss, avg_q, num_actors, etc.)
        """
        # [LOGGING ADDED] Track metrics for this training call
        training_metrics = {
            'critic_losses': [],
            'actor_losses': [],
            'q_values': [],
            'target_q_values': [],
        }
        
        # Check if found new species in archive
        diff = set(archive.keys()) - self.actors_set
        
        # [LOGGING ADDED] Log new species additions
        if len(diff) > 0 and verbose:
            print(f"    [Critic] Adding {len(diff)} new species to training pool")
        
        for desc in diff:
            # Add new species to the critic training pool
            self.actors_set.add(desc)
            new_actor = archive[desc].x
            a = copy.deepcopy(new_actor)
            for param in a.parameters():
                param.requires_grad = True
            a.parent_1_id = new_actor.id
            a.parent_2_id = None
            a.type = "critic_training"
            target = copy.deepcopy(a)
            optimizer = torch.optim.Adam(a.parameters(), lr=3e-4)
            self.actors.append(a)
            self.actor_targets.append(target)
            self.actor_optimisers.append(optimizer)

        # [LOGGING ADDED] Log training start
        if verbose:
            print(f"    [Critic] Training {nr_of_steps} steps with {len(self.actors)} actors")

        for step in range(nr_of_steps):
            self.total_it += 1
            
            # Sample replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Compute target Q-values using all actor targets
            all_target_Q = torch.zeros(batch_size, len(self.actors)).to(device)
            
            with torch.no_grad():
                # Target policy smoothing: add clipped noise to actions
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                for idx, actor_target in enumerate(self.actor_targets):
                    next_action = (
                        actor_target(next_state) + noise
                    ).clamp(-self.max_action, self.max_action)
                    
                    # Clipped double Q-learning: use min of Q1, Q2
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    all_target_Q[:, idx] = target_Q.squeeze()

                # Take max across all actors (optimistic target)
                target_Q = torch.max(all_target_Q, dim=1, keepdim=True)[0]
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss (MSE)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # [LOGGING ADDED] Track critic metrics
            training_metrics['critic_losses'].append(critic_loss.item())
            training_metrics['q_values'].append(current_Q1.mean().item())
            training_metrics['target_q_values'].append(target_Q.mean().item())

            # Delayed policy updates (every d=2 steps)
            if self.total_it % self.policy_freq == 0:
                step_actor_losses = []
                
                for idx, actor in enumerate(self.actors):
                    # Compute actor loss: maximize Q1(s, actor(s))
                    actor_loss = -self.critic.Q1(state, actor(state)).mean()

                    # Optimize the actor
                    self.actor_optimisers[idx].zero_grad()
                    actor_loss.backward()
                    self.actor_optimisers[idx].step()
                    
                    # [LOGGING ADDED] Track actor loss
                    step_actor_losses.append(actor_loss.item())

                    # Soft update actor target
                    for param, target_param in zip(actor.parameters(), self.actor_targets[idx].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # Soft update critic target
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                # [LOGGING ADDED] Track actor losses
                if step_actor_losses:
                    training_metrics['actor_losses'].append(sum(step_actor_losses) / len(step_actor_losses))

            # [LOGGING ADDED] Periodic logging during training
            if verbose and (step + 1) % 100 == 0:
                avg_critic_loss = sum(training_metrics['critic_losses'][-100:]) / min(100, len(training_metrics['critic_losses']))
                avg_q = sum(training_metrics['q_values'][-100:]) / min(100, len(training_metrics['q_values']))
                print(f"      Step {step+1}/{nr_of_steps}: "
                      f"Critic Loss={avg_critic_loss:.4f}, "
                      f"Avg Q={avg_q:.4f}")

        # [LOGGING ADDED] Compute final metrics
        final_metrics = {
            'critic_loss': sum(training_metrics['critic_losses']) / len(training_metrics['critic_losses']) if training_metrics['critic_losses'] else 0,
            'critic_loss_final': training_metrics['critic_losses'][-1] if training_metrics['critic_losses'] else 0,
            'avg_q': sum(training_metrics['q_values']) / len(training_metrics['q_values']) if training_metrics['q_values'] else 0,
            'avg_target_q': sum(training_metrics['target_q_values']) / len(training_metrics['target_q_values']) if training_metrics['target_q_values'] else 0,
            'avg_actor_loss': sum(training_metrics['actor_losses']) / len(training_metrics['actor_losses']) if training_metrics['actor_losses'] else 0,
            'num_actors': len(self.actors),
            'total_iterations': self.total_it,
        }
        
        # [LOGGING ADDED] Store for history tracking
        self.critic_losses.extend(training_metrics['critic_losses'])
        self.actor_losses.extend(training_metrics['actor_losses'])
        self.q_values.extend(training_metrics['q_values'])
        
        # [LOGGING ADDED] Print summary
        if verbose:
            print(f"    [Critic] Training complete:")
            print(f"      Avg Critic Loss: {final_metrics['critic_loss']:.4f}")
            print(f"      Avg Q-value: {final_metrics['avg_q']:.4f}")
            print(f"      Avg Actor Loss: {final_metrics['avg_actor_loss']:.4f}")
            print(f"      Actors in pool: {final_metrics['num_actors']}")

        return final_metrics

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename)
        torch.save(self.critic_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
        self.critic_target = copy.deepcopy(self.critic)
    
    # [LOGGING ADDED] Helper method to get training history
    def get_training_history(self):
        """Return full training history for plotting"""
        return {
            'critic_losses': self.critic_losses,
            'actor_losses': self.actor_losses,
            'q_values': self.q_values,
        }