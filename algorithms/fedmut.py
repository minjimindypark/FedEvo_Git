"""FedMut algorithm implementation."""
import torch
import numpy as np
from .base import BaseRunner


class FedMutRunner(BaseRunner):
    """FedMut: FedAvg with gradient-based mutation."""
    
    def __init__(self, model, clients, num_rounds, mutation_prob=0.1, device='cpu'):
        super().__init__(model, clients, num_rounds, device)
        self.mutation_prob = mutation_prob
        self.w_prev = None
        self.round_idx = 0  # Track round index for skipping mutation at r=0
        
    def run_round(self, round_idx):
        """
        Run one round of FedMut.
        
        At round 0: Only perform FedAvg aggregation (no mutation).
        At round > 0: Apply mutation using g_r = w_curr - w_prev.
        """
        client_weights = []
        client_sizes = []
        
        w_global = self.param_state_dict(self.model)
        
        # Client training
        for client in self.clients:
            self.load_param_state_dict(client.model, w_global)
            client.train()
            client_weights.append(self.param_state_dict(client.model))
            client_sizes.append(len(client.dataset))
        
        # FedAvg aggregation
        w_curr = self.fedavg_aggregate(client_weights, client_sizes)
        self.load_param_state_dict(self.model, w_curr)
        
        # Skip mutation at round 0
        if self.round_idx == 0:
            self.w_prev = {k: v.clone() for k, v in w_curr.items()}
            self.round_idx += 1
            return {}
        
        # Apply mutation for rounds > 0
        if self.w_prev is not None:
            w_mutated = self._apply_mutation(w_curr, self.w_prev)
            self.load_param_state_dict(self.model, w_mutated)
            self.w_prev = {k: v.clone() for k, v in w_mutated.items()}
        
        self.round_idx += 1
        return {}
    
    def _apply_mutation(self, w_curr, w_prev):
        """
        Apply gradient-based mutation.
        
        Mask granularity: one scalar sign per parameter tensor key from model.named_parameters().
        """
        w_mutated = {}
        
        for k in w_curr.keys():
            # Compute gradient estimate: g_r = w_curr - w_prev
            g_r = w_curr[k] - w_prev[k]
            
            # Generate mutation mask: one scalar sign per parameter tensor
            if torch.rand(1).item() < self.mutation_prob:
                # Flip the sign of the entire tensor
                sign = torch.sign(torch.randn(1)).item()
                w_mutated[k] = w_curr[k] + sign * g_r.abs()
            else:
                w_mutated[k] = w_curr[k]
        
        return w_mutated
