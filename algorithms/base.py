"""Base classes for federated learning algorithms."""
import torch
import torch.nn as nn


class BaseRunner:
    """Base runner for federated learning algorithms."""
    
    def __init__(self, model, clients, num_rounds, device='cpu'):
        self.model = model
        self.clients = clients
        self.num_rounds = num_rounds
        self.device = device
        
    def param_state_dict(self, model):
        """
        Get state dict containing only parameters (no BN buffers).
        
        BN buffers are not aggregated; only learnable parameters are aggregated.
        """
        return {k: v.cpu() for k, v in model.named_parameters()}
    
    def load_param_state_dict(self, model, state_dict):
        """Load parameters from state dict."""
        model_dict = dict(model.named_parameters())
        for k, v in state_dict.items():
            if k in model_dict:
                model_dict[k].data.copy_(v.to(self.device))
    
    def fedavg_aggregate(self, client_weights, client_sizes):
        """FedAvg aggregation."""
        total = sum(client_sizes)
        aggregated = {}
        
        for k in client_weights[0].keys():
            aggregated[k] = sum(
                w[k] * (size / total) 
                for w, size in zip(client_weights, client_sizes)
            )
        
        return aggregated
    
    def run_round(self, round_idx):
        """Run one round of federated learning."""
        raise NotImplementedError


class FedAvgRunner(BaseRunner):
    """Standard FedAvg runner."""
    
    def run_round(self, round_idx):
        """Run one round of FedAvg."""
        client_weights = []
        client_sizes = []
        
        w_global = self.param_state_dict(self.model)
        
        for client in self.clients:
            self.load_param_state_dict(client.model, w_global)
            client.train()
            client_weights.append(self.param_state_dict(client.model))
            client_sizes.append(len(client.dataset))
        
        w_new = self.fedavg_aggregate(client_weights, client_sizes)
        self.load_param_state_dict(self.model, w_new)
        
        return {}
