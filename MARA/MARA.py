import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from MARA.DropEdge import DropEdge
from config import config

class MARA(nn.Module):
    def __init__(self, simplification_type=config["simplification_type"], simplification_stages=config["simplification_stages"], simplification_strategy=config["simplification_strategy"], DE_p=config["DE_p"], NS_k=config["NS_k"], dropout=config["dropout"]):
        super().__init__()
        torch.manual_seed(1234)
        
        self.simplification_type = simplification_type
        self.simplification_stages = simplification_stages
        self.simplification_strategy = simplification_strategy
        self.DE_p = DE_p
        self.NS_k = NS_k
        
        self.conv1 = GCNConv(1000, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, 52)
        self.classifier = nn.Linear(52, 3)
        self.ReLU = torch.nn.ReLU6()

        self.dropout = torch.nn.Dropout(dropout) # ręcznie sprawdziłem dropout 0.1, 0.2 i 0.3 i żaden nie zwiększa wyniku

        self.dropedge = DropEdge(self.simplification_type, self.DE_p)

    def forward(self, x, edges, layers_lengths):
        if self.simplification_stages == "once":
            edges, layers_lengths = self.dropedge(edges, layers_lengths)
            h = self.conv1(x, edges)
            h = self.dropout(h)
            h = self.ReLU(h)
            h = self.conv2(h, edges)
            h = self.dropout(h)
            h = self.ReLU(h)
            h = self.conv3(h, edges)
            h = self.dropout(h)
            h = self.ReLU(h)

        if self.simplification_stages == "each":
            edges, layers_lengths = self.dropedge(edges, layers_lengths)
            h = self.conv1(x, edges)
            h = self.dropout(h)
            h = self.ReLU(h)
            edges, layers_lengths = self.dropedge(edges, layers_lengths)
            h = self.conv2(h, edges)
            h = self.dropout(h)
            h = self.ReLU(h)
            edges, layers_lengths = self.dropedge(edges, layers_lengths)
            h = self.conv3(h, edges)
            h = self.dropout(h)
            h = self.ReLU(h)

        out = torch.sigmoid(self.classifier(h))

        return out, h

model = MARA()
print(model)