import torch

from utils.draw_multilayer_graph import draw_multilayer_graph

def graph_info(node_classes, edges, layer_lengths):
    classes = node_classes.unique()

    edges_connecting_classes = [
        torch.sum(torch.logical_or((node_classes[edges[:,0]] == node_class), (node_classes[edges[:,1]] == node_class))).item()
        for node_class in classes
    ]

    avg_node_degree_per_layer = [
        layer_lengths[i] / node_classes.shape[0]
        for i in range(len(layer_lengths) - 1)
    ]

    avg_node_degree_per_class = [
        edges_connecting_classes[i] / torch.sum(node_classes == classes[i]).item()
        for i in range(len(classes))
    ]

    print("====== GRAPH INFO ======")
    print(f"Number of nodes: {node_classes.shape[0]}")
    print(f"Number of edges: {edges.shape[1]}")
    print(f"Number of layers: {len(layer_lengths) - 1}")
    print(f"Number of edges per layer: {layer_lengths[:-1]}")
    print(f"Number of cross-layers edges: {layer_lengths[-1]}")
    print(f"Number of nodes per class: {torch.bincount(node_classes)}")
    print(f"Number of edges connecting classes: {edges_connecting_classes}")
    print(f"Average node degree: {edges.shape[1] / node_classes.shape[0]:.2f}")
    print(f"*Average node degree per layer: {avg_node_degree_per_layer}")
    print(f"Average node degree per class: {avg_node_degree_per_class}")
    print("====== ========== ======")

    # nodes = torch.arange(node_classes.shape[0])
    # layers = torch.cat((torch.zeros(2807, dtype=torch.int), torch.ones(2807, dtype=torch.int)), dim=0)

    # print(nodes.shape)
    # print(layers.shape)
    # print(edges.shape)

    # draw_multilayer_graph(edges, nodes, "new_graph.png")
