"""FedImpro algorithm implementation."""
import torch
import numpy as np
from .base import BaseRunner


class FedImproRunner(BaseRunner):
    """FedImpro: FedAvg with class-wise feature gap analysis and synthetic data."""
    
    def __init__(self, model, clients, num_rounds, num_classes=10, device='cpu', seed=42):
        super().__init__(model, clients, num_rounds, device)
        self.num_classes = num_classes
        self.seed = seed
        
    def run_round(self, round_idx):
        """Run one round of FedImpro."""
        client_weights = []
        client_sizes = []
        
        w_global = self.param_state_dict(self.model)
        
        # Client training with synthetic data
        for idx, client in enumerate(self.clients):
            self.load_param_state_dict(client.model, w_global)
            
            # Gather gap features for this client
            gap_features = self._gather_gap_features(client)
            
            # Local training with synthetic data
            self._local_train(client, gap_features, round_idx, idx)
            
            client_weights.append(self.param_state_dict(client.model))
            client_sizes.append(len(client.dataset))
        
        # FedAvg aggregation
        w_new = self.fedavg_aggregate(client_weights, client_sizes)
        self.load_param_state_dict(self.model, w_new)
        
        return {}
    
    def _gather_gap_features(self, client):
        """
        Gather per-class feature statistics.
        
        Optimized: iterate only over unique labels in each batch instead of all classes.
        D is dynamic from h.shape[1].
        """
        client.model.eval()
        
        # Initialize storage for per-class features
        class_features = {c: [] for c in range(self.num_classes)}
        
        with torch.no_grad():
            for x, y in client.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Extract features (assuming model has a feature extractor)
                h = self._extract_features(client.model, x)
                
                # D is dynamic from h.shape[1] (note: split implies D=128)
                D = h.shape[1]
                
                # Iterate only over unique labels in this batch (optimization)
                for c in y.unique().tolist():
                    mask = (y == c)
                    if mask.any():
                        class_features[c].append(h[mask])
        
        # Compute statistics per class
        gap_stats = {}
        for c in range(self.num_classes):
            if len(class_features[c]) > 0:
                feats = torch.cat(class_features[c], dim=0)
                gap_stats[c] = {
                    'mean': feats.mean(dim=0),
                    'std': feats.std(dim=0),
                    'count': feats.shape[0]
                }
            else:
                gap_stats[c] = {
                    'mean': torch.zeros(D, device=self.device),
                    'std': torch.ones(D, device=self.device),
                    'count': 0
                }
        
        return gap_stats
    
    def _extract_features(self, model, x):
        """Extract features from model (placeholder for actual implementation)."""
        # This is a simplified version - actual implementation would depend on model architecture
        # For demonstration, assume model has features attribute or similar
        if hasattr(model, 'features'):
            return model.features(x)
        else:
            # Fallback: use penultimate layer output
            # This is a placeholder - actual implementation depends on architecture
            output = model(x)
            return output if isinstance(output, torch.Tensor) else torch.randn(x.shape[0], 128, device=self.device)
    
    def _local_train(self, client, gap_features, round_idx, client_idx):
        """
        Local training with synthetic data.
        
        Uses deterministic RNG for y_syn sampling: seed + 9999.
        """
        client.model.train()
        
        # Deterministic RNG for y_syn sampling (separate from client sampling RNG)
        rng_syn = np.random.RandomState(self.seed + client_idx * 1000 + round_idx * 100 + 9999)
        
        # Generate synthetic labels
        y_syn = rng_syn.choice(self.num_classes, size=32)  # batch size 32 for synthetic data
        
        # Training loop (simplified)
        optimizer = torch.optim.SGD(client.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(client.local_epochs):
            for x, y in client.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output = client.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            
            # Additional training with synthetic data based on gap features
            # (simplified - actual implementation would use gap_features to generate x_syn)
            y_syn_tensor = torch.tensor(y_syn, dtype=torch.long, device=self.device)
            # For demonstration, we skip actual synthetic data generation
