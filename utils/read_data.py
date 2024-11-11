import torch

def get_neighbours():
    with open("data/MARA_imdb_mlh/positives.txt", "r") as input_file:
        layer_1 = []
        layer_2 = []
        for line in input_file:
            if line == "MAM MDM\n":
                continue
            line = line.replace("\n", "").split(" ")
            if(int(line[2]) > 0):
                layer_1.append([int(line[0]), int(line[1])])
            if(int(line[3]) > 0):
                layer_2.append([int(line[0]), int(line[1])])

        return torch.tensor(layer_1), torch.tensor(layer_2)

def get_features():
    with open("data/MARA_imdb_mlh/features1000.txt", "r") as input_file:
        features = []
        for line in input_file:
            line = line.replace("\n", "").split(" ")
            features.append(list(map(float,line[2:])))
        return torch.tensor(features)
    
def get_classess():
    with open("data/MARA_imdb_mlh/classes.txt", "r") as input_file:
        classes = torch.full((2807,), -1)
        for line in input_file:
            line = line.replace("\n", "").split(" ")
            classes[int(line[0])] = int(line[1])
        return classes

class IMDB_mlh(torch.utils.data.Dataset):
    def __init__(self):
        self.node_features = get_features()
        self.nodes = torch.arange(self.node_features.shape[0])
        self.layer_1, self.layer_2 = get_neighbours()
        self.classes = get_classess()

        self.num_nodes = self.nodes.shape[0]
        self.num_edges = self.layer_1.shape[0] + self.layer_2.shape[0]
        self.layer_1_num_edges = self.layer_1.shape[0]
        self.layer_2_num_edges = self.layer_2.shape[0]
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
        print(f" Number of edges: layer1: {self.layer_1_num_edges}, layer2: {self.layer_2_num_edges}")
        print(f" Number of features: {self.num_features}")
        print(f" Number of classes: {self.num_classes}")
        print(f" Number of nodes per class: {torch.bincount(self.classes)}")