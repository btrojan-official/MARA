import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralSparse(nn.Module):
    def __init__(self, simplification_type="multilayer", k=5, input_dim=1000, hidden_dim=128, tau=1.0, anneal_steps=1000):
        super().__init__()
        self.simplification_type = simplification_type
        self.k = k
        self.tau = tau
        self.anneal_steps = anneal_steps
        self.step = 0  # Step counter for temperature annealing

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Combined node features
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, intra_layer_edges, cross_layer_edges):
        
        print("NeuralSparse input")
        print(x.shape)
        print(len(intra_layer_edges))
        print(intra_layer_edges[0].shape)
        print(len(cross_layer_edges))
        print(cross_layer_edges[0].shape)

        return intra_layer_edges, cross_layer_edges