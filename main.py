"""Main script for federated learning experiments."""
import torch
import torch.nn as nn
import numpy as np
import csv
import sys
from algorithms.base import FedAvgRunner
from algorithms.fedmut import FedMutRunner
from algorithms.fedimpro import FedImproRunner
from algorithms.fedevo import FedEvoRunner


class SimpleModel(nn.Module):
    """Simple CNN model for demonstration."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Client:
    """Federated learning client."""
    
    def __init__(self, model, dataset, local_epochs=1, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.device = device
        
        # Create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
    
    def train(self):
        """Train the model locally."""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.local_epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()


def create_dummy_dataset(num_samples=100, num_classes=10):
    """Create a dummy dataset for demonstration."""
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, num_classes, (num_samples,))
    return torch.utils.data.TensorDataset(X, y)


def evaluate(model, test_loader, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return 100 * correct / total if total > 0 else 0.0


def compute_train_loss(model, train_loader, device='cpu'):
    """Compute average training loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def estimate_uplink_bytes(model):
    """Estimate uplink bytes for transmitting model parameters."""
    total_bytes = 0
    for param in model.parameters():
        # Assume float32 (4 bytes per parameter)
        total_bytes += param.numel() * 4
    return total_bytes


def main(algorithm='fedavg', num_rounds=10, num_clients=5, seed=42):
    """
    Main function for federated learning.
    
    BN buffers policy:
    - BN buffers are not aggregated; only learnable parameters are aggregated.
    - Transmission and aggregation use named_parameters() only.
    - BN running stats (buffers) remain local to each client.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = SimpleModel(num_classes=10).to(device)
    
    # Create clients with dummy data
    clients = []
    for i in range(num_clients):
        client_model = SimpleModel(num_classes=10).to(device)
        client_dataset = create_dummy_dataset(num_samples=100)
        clients.append(Client(client_model, client_dataset, local_epochs=1, device=device))
    
    # Create test dataset
    test_dataset = create_dummy_dataset(num_samples=200)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create train dataset for loss computation
    train_dataset = create_dummy_dataset(num_samples=500)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # Initialize runner based on algorithm
    if algorithm == 'fedavg':
        runner = FedAvgRunner(model, clients, num_rounds, device=device)
    elif algorithm == 'fedmut':
        runner = FedMutRunner(model, clients, num_rounds, mutation_prob=0.1, device=device)
    elif algorithm == 'fedimpro':
        runner = FedImproRunner(model, clients, num_rounds, num_classes=10, device=device, seed=seed)
    elif algorithm == 'fedevo':
        runner = FedEvoRunner(model, clients, num_rounds, pop_size=5, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # CSV output
    csv_writer = csv.writer(sys.stdout)
    csv_writer.writerow(['round', 'test_accuracy', 'train_loss', 'uplink_bytes'])
    
    # Training loop
    for round_idx in range(num_rounds):
        # Run one round
        runner.run_round(round_idx)
        
        # Evaluate
        if isinstance(runner, FedEvoRunner):
            # For FedEvo: evaluate runner.pop[0], which is the elite (highest-ranked by usage count)
            # after _evolve() has sorted the population in run_round().
            eval_model = runner.pop[0]
        else:
            eval_model = runner.model
        
        test_acc = evaluate(eval_model, test_loader, device=device)
        train_loss = compute_train_loss(eval_model, train_loader, device=device)
        uplink_bytes = estimate_uplink_bytes(eval_model) * num_clients  # Total uplink from all clients
        
        # Output CSV row
        csv_writer.writerow([round_idx, f'{test_acc:.2f}', f'{train_loss:.4f}', uplink_bytes])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='fedavg', 
                       choices=['fedavg', 'fedmut', 'fedimpro', 'fedevo'])
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    main(algorithm=args.algorithm, num_rounds=args.num_rounds, 
         num_clients=args.num_clients, seed=args.seed)
