config = {
    "simplification_type": "multilayer", # ["l-b-l","multilayer"]
    "simplification_stages":"once", # ["once","each"]
    "simplification_strategy":"NS", # ["DE","NS"] (DropEdge/NeuralSparse)
    "DE_p":0.4,
    "NS_k":1,
    "dropout":0.001,
    "input_dim": 1000,

    "lr":1.5e-3,
    "weight_decay":4e-4,
    "epoch_num": 250,
    "patience": 10
}