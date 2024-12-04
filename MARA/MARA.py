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

        self.neuralsparse_1 = NeuralSparse(self.simplification_type, self.DE_p) # self.NS_k
        self.neuralsparse_2 = NeuralSparse(self.simplification_type, self.DE_p) # self.NS_k
        self.neuralsparse_3 = NeuralSparse(self.simplification_type, self.DE_p) # self.NS_k

    def forward(self, x, edges, layers_lengths):

        # layer lengths are numbers of edges in each layer starting from layer 0 and last element is number of cross-layer edges
        if self.simplification_strategy == "DE":
            if self.simplification_stages == "once":
                edges, layers_lengths = self.dropedge(edges, layers_lengths)
                h = self.ReLU(self.dropout(self.conv1(x, edges)))
                h = self.ReLU(self.dropout(self.conv2(h, edges)))
                h = self.ReLU(self.dropout(self.conv3(h, edges)))

            if self.simplification_stages == "each":
                edges, layers_lengths = self.dropedge(edges, layers_lengths)
                h = self.ReLU(self.dropout(self.conv1(x, edges)))
                edges, layers_lengths = self.dropedge(edges, layers_lengths)
                h = self.ReLU(self.dropout(self.conv2(h, edges)))
                edges, layers_lengths = self.dropedge(edges, layers_lengths)
                h = self.ReLU(self.dropout(self.conv3(h, edges)))

        elif self.simplification_strategy == "NS":
            if self.simplification_stages == "once":
                edges, layers_lengths = self.neuralsparse_1(x, edges, layers_lengths)
                h = self.conv1(x, edges)
                h = self.dropout(h)
                h = self.ReLU(h)
                h = self.conv2(h, edges)
                h = self.dropout(h)
                h = self.ReLU(h)
                h = self.conv3(h, edges)
                h = self.dropout(h)
                h = self.ReLU(h)

        #     if self.simplification_stages == "each":
        #         edges, layers_lengths = self.neuralsparse_1(x, edges, layers_lengths)
        #         h = self.conv1(x, edges)
        #         h = self.dropout(h)
        #         h = self.ReLU(h)
        #         edges, layers_lengths = self.neuralsparse_2(x, edges, layers_lengths)
        #         h = self.conv2(h, edges)
        #         h = self.dropout(h)
        #         h = self.ReLU(h)
        #         edges, layers_lengths = self.neuralsparse_3(x, edges, layers_lengths)
        #         h = self.conv3(h, edges)
        #         h = self.dropout(h)

        out = torch.sigmoid(self.classifier(h))

        return out, edges, layers_lengths

model = MARA()
print(model)