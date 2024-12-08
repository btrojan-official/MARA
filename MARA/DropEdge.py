import torch
import torch.nn as nn

class DropEdge(nn.Module):
    def __init__(self, simplification_type="l-b-l", p=0.2):
        super().__init__()
        self.simplification_type = simplification_type
        self.p = p
        
    def forward(self, intra_layer_edges, cross_layer_edges):
        if(self.simplification_type == "l-b-l"):
            new_intra_layer_edges = []

            for i in range(len(intra_layer_edges)):
                intra_mask = torch.rand(intra_layer_edges[i].size(0)) > self.p

                new_intra_layer_edges.append(intra_layer_edges[i][intra_mask,:])

            return new_intra_layer_edges, cross_layer_edges
        
        if(self.simplification_type == "multilayer"):
            new_intra_layer_edges = []
            new_cross_layer_edges = []

            for i in range(len(intra_layer_edges)):
                intra_mask = torch.rand(intra_layer_edges[i].size(0)) > self.p

                new_intra_layer_edges.append(intra_layer_edges[i][intra_mask,:])

            for i in range(len(cross_layer_edges)):
                cross_mask = torch.rand(cross_layer_edges[i].size(0)) > self.p

                new_cross_layer_edges.append(cross_layer_edges[i][cross_mask,:])

            return new_intra_layer_edges, new_cross_layer_edges
        




    # def forward(self, edges, layers_lengths):
    #     if(self.simplification_type == "l-b-l"):
    #         intra_layers_length = torch.sum(layers_lengths[:-1])
    #         intra_mask = torch.rand(intra_layers_length) > self.p

    #         intra_layers = edges[:,:intra_layers_length]
    #         edges = torch.cat([intra_layers[:,intra_mask], edges[:,intra_layers_length:]], dim=1)

    #         new_layers_lenghts = []
    #         temp = 0
    #         for i in range(len(layers_lengths)-1):
    #             new_layers_lenghts.append(torch.sum(intra_mask[temp:temp + layers_lengths[i]]))
    #             temp += layers_lengths[i]
    #         new_layers_lenghts.append(layers_lengths[-1])

    #         return edges, torch.tensor(new_layers_lenghts)
        
    #     if(self.simplification_type == "multilayer"):
    #         mask = torch.rand(edges.shape[1]) > self.p
    #         edges = edges[:, mask]

    #         new_layers_lenghts = []
    #         temp = 0
    #         for i in range(len(layers_lengths)):
    #             new_layers_lenghts.append(torch.sum(mask[temp:temp + layers_lengths[i]]))
    #             temp += layers_lengths[i]

    #         return edges, torch.tensor(new_layers_lenghts)