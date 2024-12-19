import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralSparse(nn.Module):
    def __init__(self, simplification_type="multilayer", k=5, input_dim=1000, hidden_dim=128, tau=1.0, anneal_steps=1000):
        super().__init__()
        self.simplification_type = simplification_type
        self.k = k
        self.tau = tau

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, intra_layer_edges, cross_layer_edges, node_layers):
        # muszis cały kod zrefaktoryzować tak, by node_layers zwracała klasa datasetu oraz by odpowiednio wrzucać to do MARY

        # sprawdź l-b-l, czy działa poprawnie

        # To check if weights are being trained
        # for idx, layer in enumerate(self.mlp):
        #     if isinstance(layer, nn.Linear): 
        #         print(f"Layer {idx} weights: {layer.weight.data[0,:10]}")

        node_edge_map = {}
        node_edge_map = self.add_edges(intra_layer_edges, node_edge_map)
        if self.simplification_type == "multilayer":
            node_edge_map = self.add_edges(cross_layer_edges, node_edge_map)

        node_weights = {}
        for node in node_edge_map.keys():
            node_scores = []
            for edge in node_edge_map[node]:
                edge_score = self.mlp(torch.cat((x[edge[0],:], x[edge[1],:]), dim=0))
                node_scores.append(edge_score)

            gumbel_softmax_weights = F.gumbel_softmax(torch.cat(node_scores, dim=0), tau=self.tau, hard=True).to(x.device)
            for _ in range(self.k-1):
                # dodaj tutaj maskowanie wcześniej wybranych node-ów
                gumbel_softmax_weights += F.gumbel_softmax(torch.cat(node_scores, dim=0), tau=self.tau, hard=True).to(x.device)

            node_weights[node] = {
                "scores": gumbel_softmax_weights,
                "edges": node_edge_map[node]
            }

        layers_ids = torch.unique(node_layers, sorted=True)
        num_of_layers = layers_ids.shape[0]

        new_intra_layer_edges = [[] for _ in range(num_of_layers)]
        new_cross_layer_edges = []
        intra_layer_weights = [[] for _ in range(num_of_layers)]
        cross_layer_weights = []
        
        for node in node_weights.keys():
            for i in range(len(node_weights[node]["edges"])):
                edge = node_weights[node]["edges"][i]
                edge_score = node_weights[node]["scores"][i]

                if(node_layers[edge[0]] == node_layers[edge[1]]):
                    id = torch.where(node_layers[edge[0]] == layers_ids)[0]

                    new_intra_layer_edges[id].append(edge)
                    intra_layer_weights[id].append(edge_score)
                else:
                    new_cross_layer_edges.append(edge)
                    cross_layer_weights.append(edge_score)

        for i in range(len(new_intra_layer_edges)):
            new_intra_layer_edges[i] = torch.tensor(new_intra_layer_edges[i]).to(x.device)
            intra_layer_weights[i] = torch.stack(intra_layer_weights[i]).to(x.device)
    
        if self.simplification_type == "multilayer":
            return new_intra_layer_edges, [torch.tensor(new_cross_layer_edges).to(x.device)], intra_layer_weights, [torch.tensor(cross_layer_weights).to(x.device)]
        return new_intra_layer_edges, cross_layer_edges, intra_layer_weights, torch.ones((cross_layer_weights[0]))
    def add_edges(self,edge_list, node_edge_map):
        for edge_tensor in edge_list:
            for edge in edge_tensor:
                node1, node2 = edge.tolist()

                if node1 not in node_edge_map:
                    node_edge_map[node1] = []
                node_edge_map[node1].append((node1, node2))

        return node_edge_map