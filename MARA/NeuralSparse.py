import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralSparse(nn.Module):
    def __init__(self, simplification_type="l-b-l", k=5, input_dim=1000, hidden_dim=128, tau_start=1.0, tau_end=0.1, anneal_steps=1000):
        super().__init__()
        self.simplification_type = simplification_type
        self.k = k
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps
        self.step = 0  # Step counter for temperature annealing

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Combined node features
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def anneal_temperature(self):
        """Anneal the temperature for Gumbel-Softmax."""
        self.step += 1
        progress = min(self.step / self.anneal_steps, 1.0)
        tau = self.tau_start + progress * (self.tau_end - self.tau_start)
        return tau

    def forward(self, node_features, edges, layer_lengths):
        """
        Args:
            node_features: Tensor of shape (N, F) where N is the number of nodes, F is the feature dimension.
            edges: List of edges (u, v) representing connections in the graph.
        
        Returns:
            selected_edges: List of sampled edges after sparsification.
        """

        if self.simplification_type == "l-b-l":
            return edges

        selected_edges = []

        # Prepare data for sparsification
        node_features = node_features.to(torch.float32)
        adjacency_matrix = adjacency_matrix.to(torch.float32)
        tau = self.anneal_temperature()

        for u in range(node_features.shape[0]):  # Iterate over nodes
            # Collect one-hop neighbors of node u
            neighbors = torch.nonzero(adjacency_matrix[u], as_tuple=True)[0]
            if len(neighbors) == 0:
                continue

            # Compute z_u,v for each neighbor v
            z_values = []
            for v in neighbors:
                edge_features = adjacency_matrix[u, v].unsqueeze(0)
                input_features = torch.cat([node_features[u], node_features[v], edge_features], dim=0)
                z_values.append(self.mlp(input_features).squeeze(0))
            z_values = torch.stack(z_values)

            # Compute probabilities π_u,v
            pi_values = F.softmax(z_values, dim=0)

            # Sample edges using Gumbel-Softmax
            for _ in range(self.k):
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(pi_values)))
                gumbel_scores = (torch.log(pi_values) + gumbel_noise) / tau
                gumbel_softmax = F.softmax(gumbel_scores, dim=0)

                # Pick the edge with the highest gumbel_softmax value
                selected_neighbor_idx = torch.argmax(gumbel_softmax)
                selected_edges.append((u, neighbors[selected_neighbor_idx].item()))

        return selected_edges, adjacency_matrix
