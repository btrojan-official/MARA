import torch
import torch.nn as nn

class NeuralSparse(nn.Module):
    def __init__(self, simplification_type="l-b-l", k=5):
        super().__init__()
        self.simplification_type = simplification_type
        self.k = k

        self.network = nn.Linear(2000, 1)

    def forward(self, node_feratures, edges, layers_lengths):
        if(self.simplification_type == "l-b-l"):
            return edges, layers_lengths
        
        if(self.simplification_type == "multilayer"):
            return edges, layers_lengths