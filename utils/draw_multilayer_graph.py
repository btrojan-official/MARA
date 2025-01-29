import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_multilayer_graph(edges_tensor, class_tensor, filename='graph.png'):
    G = nx.Graph()
    
    for edge in edges_tensor:
        G.add_edge(edge[0], edge[1])

    node_colors = []
    for node in G.nodes():
        node_class = class_tensor[node]  
        node_colors.append(node_class)

    layout = nx.spring_layout(G, k=0.15, iterations=20)
    
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos=layout, node_size=50, node_color=node_colors, cmap=plt.cm.viridis, with_labels=False, alpha=0.7)

    plt.savefig(filename, format='PNG')
    
    plt.show()

# Example usage:
# Assuming `edges_tensor` is of shape (14000, 2) and `class_tensor` is of shape (5600,)
# Generate random edges and node classes for the example

# # For demonstration, let's simulate data
# edges_tensor = np.random.randint(0, 5600, size=(14000, 2)) 
# class_tensor = np.random.randint(0, 10, size=(5600,)) 

# plot_and_save_graph(edges_tensor, class_tensor, filename='graph_output.png')
