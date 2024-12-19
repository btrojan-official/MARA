import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from MARA.DropEdge import DropEdge
from MARA.NeuralSparse import NeuralSparse

from config import config

class MARA(nn.Module):
    def __init__(self, simplification_type=config["simplification_type"], simplification_stages=config["simplification_stages"], simplification_strategy=config["simplification_strategy"], DE_p=config["DE_p"], NS_k=config["NS_k"], dropout=config["dropout"], input_dim=config["input_dim"]):
        super().__init__()
        torch.manual_seed(1234)
        
        self.simplification_type = simplification_type
        self.simplification_stages = simplification_stages
        self.simplification_strategy = simplification_strategy
        self.DE_p = DE_p
        self.NS_k = NS_k
        self.input_dim = input_dim
        
        self.conv1 = GCNConv(self.input_dim, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, 52)
        self.classifier = nn.Linear(52, 3)
        self.ReLU = torch.nn.ReLU6()

        self.dropout = torch.nn.Dropout(dropout)

        self.dropedge = DropEdge(self.simplification_type, self.DE_p)

        self.neuralsparse_1 = NeuralSparse(self.simplification_type, self.NS_k)
        self.neuralsparse_2 = NeuralSparse(self.simplification_type, self.NS_k)
        self.neuralsparse_3 = NeuralSparse(self.simplification_type, self.NS_k)

    def forward(self, x, intra_layer_edges, cross_layer_edges):

        # layer lengths are numbers of edges in each layer starting from layer 0 and last element is number of cross-layer edges
        if self.simplification_strategy == "DE":
            if self.simplification_stages == "once":
                intra_layer_edges, cross_layer_edges = self.dropedge(intra_layer_edges, cross_layer_edges)
                h = self.ReLU(self.dropout(self.conv1(x, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T)))
                h = self.ReLU(self.dropout(self.conv2(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T)))
                h = self.ReLU(self.dropout(self.conv3(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T)))

            if self.simplification_stages == "each":
                intra_layer_edges, cross_layer_edges = self.dropedge(intra_layer_edges, cross_layer_edges)
                h = self.ReLU(self.dropout(self.conv1(x, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T)))
                intra_layer_edges, cross_layer_edges = self.dropedge(intra_layer_edges, cross_layer_edges)
                h = self.ReLU(self.dropout(self.conv2(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T)))
                intra_layer_edges, cross_layer_edges = self.dropedge(intra_layer_edges, cross_layer_edges)
                h = self.ReLU(self.dropout(self.conv3(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T)))

        elif self.simplification_strategy == "NS":
            if self.simplification_stages == "once":
                intra_layer_edges, cross_layer_edges, intra_layer_weights, cross_layer_weights = self.neuralsparse_1(x, intra_layer_edges, cross_layer_edges, torch.cat((torch.zeros(len(x)//2), torch.ones(len(x)//2))).to(x.device))
                h = self.ReLU(self.dropout(self.conv1(x, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weights=torch.cat(intra_layer_weights + cross_layer_weights, dim=0).T)))
                h = self.ReLU(self.dropout(self.conv2(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weights=torch.cat(intra_layer_weights + cross_layer_weights, dim=0).T)))
                h = self.ReLU(self.dropout(self.conv3(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weights=torch.cat(intra_layer_weights + cross_layer_weights, dim=0).T)))

        out = torch.sigmoid(self.classifier(h))

        return out, intra_layer_edges, cross_layer_edges

model = MARA()
print(model)