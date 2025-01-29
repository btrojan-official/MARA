import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from config import config
from MARA.DropEdge import DropEdge
from MARA.NeuralSparse import NeuralSparse


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
        self.hidden_size_conv1 = 512
        self.hidden_size_conv2 = 256
        self.hidden_size_conv3 = 52
        self.num_classes = 3
        
        self.conv1 = GCNConv(self.input_dim, self.hidden_size_conv1)
        self.conv2 = GCNConv(self.hidden_size_conv1, self.hidden_size_conv2)
        self.conv3 = GCNConv(self.hidden_size_conv2, self.hidden_size_conv3)
        self.classifier = nn.Linear(self.hidden_size_conv3, self.num_classes)
        self.ReLU = torch.nn.ReLU6()

        self.dropout = torch.nn.Dropout(dropout)
        if simplification_strategy == "DE":
            self.dropedge = DropEdge(self.simplification_type, self.DE_p)
        if self.simplification_strategy == "NS":
            self.neuralsparse_1 = NeuralSparse(self.simplification_type, self.NS_k, input_dim=self.input_dim)
            self.neuralsparse_2 = NeuralSparse(self.simplification_type, self.NS_k, input_dim=self.hidden_size_conv1) # self.NS_k
            self.neuralsparse_3 = NeuralSparse(self.simplification_type, self.NS_k, input_dim=self.hidden_size_conv2) # self.NS_k

    def forward(self, x, node_layers, intra_layer_edges, cross_layer_edges):
        """
        x: node features torch.tensor[number_of_nodes, node_vector_dim]
        node_layers: node layers torch.tensor[number_of_nodes, 1]
        intra_layer_edges: intra layer edges list List[torch(tensor[number_of_edges, 2]), ...]
        cross_layer_edges: cross layer edges list List[torch(tensor[number_of_edges, 2])]

        return:
        out: new node features torch.tensor[number_of_nodes, node_new_vector_dim]
        intra_layer_edges: intra layer edges list List[torch(tensor[new_number_of_edges, 2]), ...]
        cross_layer_edges: cross layer edges list List[torch(tensor[new_number_of_edges, 2])]
        """

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
                intra_layer_edges, cross_layer_edges, intra_layer_weights, cross_layer_weights = self.neuralsparse_1(x, intra_layer_edges, cross_layer_edges, node_layers)
                h = self.ReLU(self.dropout(self.conv1(x, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weight=torch.cat(intra_layer_weights + cross_layer_weights, dim=0))))
                h = self.ReLU(self.dropout(self.conv2(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weight=torch.cat(intra_layer_weights + cross_layer_weights, dim=0))))
                h = self.ReLU(self.dropout(self.conv3(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weight=torch.cat(intra_layer_weights + cross_layer_weights, dim=0))))

            if self.simplification_stages == "each":
                intra_layer_edges, cross_layer_edges, intra_layer_weights_1, cross_layer_weights_1 = self.neuralsparse_1(x, intra_layer_edges, cross_layer_edges, node_layers)
                h = self.ReLU(self.dropout(self.conv1(x, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weight=torch.cat(intra_layer_weights_1 + cross_layer_weights_1, dim=0))))
                
                intra_layer_masked_1 = [intra_layer_edges[i][intra_layer_weights_1[i].bool()] for i in range(len(intra_layer_edges))]
                cross_layer_masked_1 = [cross_layer_edges[i][cross_layer_weights_1[i].bool()] for i in range(len(cross_layer_edges))]
                intra_layer_edges, cross_layer_edges, intra_layer_weights_2, cross_layer_weights_2 = self.neuralsparse_2(h, intra_layer_masked_1, cross_layer_masked_1, node_layers)
                h = self.ReLU(self.dropout(self.conv2(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weight=torch.cat(intra_layer_weights_2 + cross_layer_weights_2, dim=0))))
                
                intra_layer_masked_2 = [intra_layer_edges[i][intra_layer_weights_2[i].bool()] for i in range(len(intra_layer_edges))]
                cross_layer_masked_2 = [cross_layer_edges[i][cross_layer_weights_2[i].bool()] for i in range(len(cross_layer_edges))]
                intra_layer_edges, cross_layer_edges, intra_layer_weights, cross_layer_weights = self.neuralsparse_3(h, intra_layer_masked_2, cross_layer_masked_2, node_layers)
                h = self.ReLU(self.dropout(self.conv3(h, torch.cat(intra_layer_edges + cross_layer_edges, dim=0).T, edge_weight=torch.cat(intra_layer_weights + cross_layer_weights, dim=0))))

        out = torch.sigmoid(self.classifier(h))

        if self.simplification_strategy == "NS":
            intra_layer_masked = []
            cross_layer_masked = [cross_layer_edges[0][cross_layer_weights[0].bool()]]

            for i in range(len(intra_layer_edges)):
                intra_layer_masked.append(intra_layer_edges[i][intra_layer_weights[i].bool()])

            return out, intra_layer_masked, cross_layer_masked
        
        return out, intra_layer_edges, cross_layer_edges

model = MARA()
print(model)