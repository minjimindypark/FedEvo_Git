"""FedEvo algorithm implementation."""
import torch
import numpy as np
from .base import BaseRunner


class FedEvoRunner(BaseRunner):
    """FedEvo: Evolutionary federated learning with population of models."""
    
    def __init__(self, model, clients, num_rounds, pop_size=5, device='cpu'):
        super().__init__(model, clients, num_rounds, device)
        self.pop_size = pop_size
        self.pop = [self._clone_model(model) for _ in range(pop_size)]
        self.usage = [0] * pop_size  # Track usage of each candidate
        self.usage_round = [0] * pop_size  # Track usage in current round
        self.last_eval_candidate_idx = 0  # Store index of candidate to evaluate
        
    def _clone_model(self, model):
        """Clone a model."""
        import copy
        return copy.deepcopy(model)
    
    def run_round(self, round_idx):
        """
        Run one round of FedEvo.
        
        Calls _evolve() which sorts elites by usage count first.
        """
        client_weights = []
        client_sizes = []
        
        # Reset round usage
        self.usage_round = [0] * self.pop_size
        
        # Each client selects a candidate
        for client in self.clients:
            # Select candidate (simplified: uniform random selection)
            candidate_idx = np.random.randint(0, self.pop_size)
            
            # Track usage
            self.usage[candidate_idx] += 1
            self.usage_round[candidate_idx] += 1
            
            # Train with selected candidate
            w_candidate = self.param_state_dict(self.pop[candidate_idx])
            self.load_param_state_dict(client.model, w_candidate)
            client.train()
            
            client_weights.append(self.param_state_dict(client.model))
            client_sizes.append(len(client.dataset))
        
        # Update population with aggregated weights
        w_new = self.fedavg_aggregate(client_weights, client_sizes)
        
        # Simple evolution: update one candidate with new weights
        update_idx = np.random.randint(0, self.pop_size)
        self.load_param_state_dict(self.pop[update_idx], w_new)
        
        # Evolve population: sort by usage count (elites first)
        self._evolve()
        
        # Update main model with elite (pop[0] after evolution)
        self.load_param_state_dict(self.model, self.param_state_dict(self.pop[0]))
        
        return {}
    
    def _evolve(self):
        """
        Evolve population by sorting candidates by usage count (elites first).
        
        After this call, pop[0] contains the candidate with the highest usage count.
        """
        # Sort population by usage count (descending order)
        sorted_indices = sorted(range(self.pop_size), key=lambda i: self.usage[i], reverse=True)
        self.pop = [self.pop[i] for i in sorted_indices]
        self.usage = [self.usage[i] for i in sorted_indices]
