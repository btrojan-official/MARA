import torch

from utils.draw_multilayer_graph import draw_multilayer_graph


def graph_info(node_classes, node_layers, intra_edges, cross_edges):
    edges_per_layer = [intra_edges[i].shape[0] for i in range(len(intra_edges))]
    edges_cross_layer = [cross_edges[i].shape[0] for i in range(len(cross_edges))]
    layer_lengths = edges_per_layer + edges_cross_layer
    edges = torch.cat(intra_edges + cross_edges, dim=0).T

    classes = node_classes.unique()

    edges_connecting_classes = [
        torch.sum(torch.logical_or((node_classes[edges[0,:]] == node_class), (node_classes[edges[1,:]] == node_class))).item()
        for node_class in classes
    ]

    avg_node_degree_per_layer = [
        layer_lengths[i] / node_layers[node_layers==i].shape[0]
        for i in range(len(layer_lengths) - 1)
    ]

    avg_node_degree_per_class = [
        edges_connecting_classes[i] / torch.sum(node_classes == classes[i]).item()
        for i in range(len(classes))
    ]

    print("\n====== GRAPH INFO ======")
    print(f"Number of nodes: {node_classes.shape[0]}")
    print(f"Number of edges: {edges.shape[1]}")
    print(f"Number of layers: {len(edges_per_layer)}")
    print(f"Number of edges per layer: {edges_per_layer}")
    print(f"Number of cross-layers edges: {edges_cross_layer}")
    print(f"Number of nodes per class: {torch.bincount(node_classes)}")
    print(f"Number of edges connecting classes: {edges_connecting_classes}")
    print(f"Average node degree: {edges.shape[1] / node_classes.shape[0]:.2f}")
    print(f"Average node degree per layer: {avg_node_degree_per_layer}")
    print(f"Average node degree per class: {avg_node_degree_per_class}")
    print("====== ========== ======\n")
