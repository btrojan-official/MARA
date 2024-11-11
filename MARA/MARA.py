from config import config
import torch

class MARA():
    def __init__(self, simplificaton_type=config["simplification_type"], simplification_stages=config["simplification_stages"], simplification_strategy=config["simplification_strategy"], DE_p=config["DE_p"], NS_k=config["NS_k"]):
        self.simplification_type = simplificaton_type
        self.simplification_stages = simplification_stages
        self.simplification_strategy = simplification_strategy
        self.DE_p = DE_p
        self.NS_k = NS_k

    def simplify(self, nodes_for_each_layer, edges_for_each_layer, cross_layer_edges, node_classes):
        if(self.simplification_strategy == "DE"):
            if(self.simplification_type == "l-b-l"):
                simplified = []
                for layer in range(len(edges_for_each_layer)):
                    print(edges_for_each_layer[layer].shape)
                    mask = torch.rand(1, edges_for_each_layer[layer].shape[0]) > self.DE_p
                    simplified.append(edges_for_each_layer[layer][mask.squeeze()].clone())
                    print(simplified[layer].shape)
                return simplified

