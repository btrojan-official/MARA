import torch
import dgl
def get_neighbours():
    MAM0_src, MAM0_dst = dgl.load_graphs("data/imdb_mlh/prep_data/MAM_0.bin")[0][0].edges()
    MAM1_src, MAM1_dst = dgl.load_graphs("data/imdb_mlh/prep_data/MAM_1.bin")[0][0].edges()
    MAM01_src, MAM01_dst = dgl.load_graphs("data/imdb_mlh/prep_data/MAM_01.bin")[0][0].edges()

    layer_1_src = torch.cat((MAM0_src, MAM1_src, MAM01_src), dim=0)
    layer_1_dst = torch.cat([MAM0_dst, MAM1_dst, MAM01_dst], dim=0)

    layer_1 = torch.stack((layer_1_src, layer_1_dst), dim=1)

    MDM0_src, MDM0_dst = dgl.load_graphs("data/imdb_mlh/prep_data/MDM_0.bin")[0][0].edges()
    MDM1_src, MDM1_dst = dgl.load_graphs("data/imdb_mlh/prep_data/MDM_1.bin")[0][0].edges()
    MDM01_src, MDM01_dst = dgl.load_graphs("data/imdb_mlh/prep_data/MDM_01.bin")[0][0].edges()

    layer_2_src = torch.cat((MDM0_src, MDM1_src, MDM01_src), dim=0)
    layer_2_dst = torch.cat([MDM0_dst, MDM1_dst, MDM01_dst], dim=0)

    layer_2 = torch.stack((layer_2_src, layer_2_dst), dim=1) + 2807

    cross_edges_1 = torch.stack((torch.arange(0,2807), torch.arange(2807,5614)),dim=1)
    cross_edges_2 = torch.stack((torch.arange(2807,5614), torch.arange(0,2807)),dim=1)
    cross_edges = torch.cat((cross_edges_1, cross_edges_2), dim=0)
    
    return layer_1, layer_2, cross_edges

def get_features():
    with open("data/imdb_mlh/features1000.txt", "r") as input_file:
        features = []
        for line in input_file:
            line = line.replace("\n", "").split(" ")
            features.append(list(map(float,line[2:])))
        return torch.tensor(features)
    
def get_classess():
    with open("data/imdb_mlh/classes.txt", "r") as input_file:
        classes = torch.full((2807,), -1)
        for line in input_file:
            line = line.replace("\n", "").split(" ")
            classes[int(line[0])] = int(line[1])
        return classes

class IMDB_mlh(torch.utils.data.Dataset):
    def __init__(self):
        self.node_features = get_features()
        self.nodes = torch.arange(self.node_features.shape[0])
        self.layer_1, self.layer_2, self.cross_edges = get_neighbours()
        self.classes = get_classess()

        self.num_nodes = self.nodes.shape[0]*2
        self.num_edges = self.layer_1.shape[0] + self.layer_2.shape[0] + self.cross_edges.shape[0]
        self.layer_1_num_edges = self.layer_1.shape[0]
        self.layer_2_num_edges = self.layer_2.shape[0]
        self.cross_edges_num = self.cross_edges.shape[0]
        self.num_features = self.node_features.shape[1]
        self.num_classes = self.classes.unique().shape[0]

    def __len__(self):
        return self.nodes.shape[0]
    
    def __getitem__(self, idx):
        return self.node_features[idx], self.classes[idx]
    
    def get_training_mask(self, mask_size=0.2):
        random_tensor = torch.rand(self.nodes.shape[0])
        return random_tensor > (1-mask_size)
    
    def info(self):
        print(f"IMDB movie type dataset:")
        print(f" Number of nodes: {self.num_nodes}")
        print(f" Number of edges: layer1: {self.layer_1_num_edges}, layer2: {self.layer_2_num_edges}, cross_layer: {self.cross_edges_num}")
        print(f" Number of features: {self.num_features}")
        print(f" Number of classes: {self.num_classes}")
        print(f" Number of nodes per class: {torch.bincount(self.classes)*2}")