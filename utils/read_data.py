import dgl
import torch


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
        features = torch.tensor(features)
        return torch.cat((features,features),dim=0)
    
def get_classess():
    with open("data/imdb_mlh/classes.txt", "r") as input_file:
        classes = torch.full((2807,), -1)
        for line in input_file:
            line = line.replace("\n", "").split(" ")
            classes[int(line[0])] = int(line[1])
        return torch.cat((classes,classes),dim=0)
    
def get_layers():
    return torch.cat((torch.zeros(len(get_classess())//2), torch.ones(len(get_classess())//2)))

class IMDB_mlh(torch.utils.data.Dataset):
    def __init__(self):
        self.node_features = get_features()
        self.nodes = torch.arange(self.node_features.shape[0])
        self.layer_1, self.layer_2, self.cross_edges = get_neighbours()
        self.classes = get_classess()
        self.node_layers = get_layers()

        self.device = "cpu"

    def __len__(self):
        return self.nodes.shape[0]*2
    
    def __getitem__(self, idx):
        return self.node_features[idx], self.classes[idx]
    
    def get_training_mask(self, train_mask_size=0.25, val_mask_size=0.25):
        random_tensor = torch.rand(self.nodes.shape[0]//2).to(self.device)
        train_mask = (random_tensor > (1-train_mask_size)).repeat(2).to(self.device)
        val_mask = torch.logical_and((random_tensor > (1-train_mask_size-val_mask_size)).repeat(2), ~train_mask).to(self.device)
        test_mask = torch.logical_and(~train_mask,~val_mask).to(self.device)

        return train_mask, val_mask, test_mask

    def get_number_of_nodes(self):
        return self.nodes.shape[0]
    
    def get_number_of_edges(self):
        return self.layer_1.shape[0] + self.layer_2.shape[0] + self.cross_edges.shape[0]
    
    def get_number_of_features(self):
        return self.node_features.shape[1]
    
    def get_number_of_classes(self):
        return self.classes.unique().shape[0]

    def info(self):
        print(f"IMDB movie type dataset:")
        print(f" Number of nodes: {self.get_number_of_nodes()}")
        print(f" Number of edges: {self.get_number_of_edges()}")
        print(f" Number of edges: layer1: {self.layer_1.shape[0]}, layer2: {self.layer_2.shape[0]}, cross_layer: {self.cross_edges.shape[0]}")
        print(f" Number of features: {self.get_number_of_features()}")
        print(f" Number of classes: {self.get_number_of_classes()}")
        print(f" Number of nodes per class: {torch.bincount(self.classes)}\n")

    def to(self, device):
        self.device = device

        self.node_features = self.node_features.to(device)
        self.nodes = self.nodes.to(device)
        self.layer_1 = self.layer_1.to(device)
        self.layer_2 = self.layer_2.to(device)
        self.cross_edges = self.cross_edges.to(device)
        self.classes = self.classes.to(device)
        return self