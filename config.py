config = {
    "simplification_type":"l-b-l", # ["l-b-l","multilayer"]
    "simplification_stages":"each", # ["once","each"]
    "simplification_strategy":"DE", # ["DE","NS"] (DropEdge/NeuralSparse)
    "DE_p":0.4,
    "NS_k":5,
    "dropout":0.001,
    "input_dim": 1000,

    "lr":1.5e-3,
    "weight_decay":4e-4,
    "epoch_num": 250,
    "patience": 50
}