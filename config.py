config = {
    "simplification_type":"l-b-l", # ["l-b-l","multilayer"]
    "simplification_stages":"each", # ["once","each"]
    "simplification_strategy":"NS", # ["DE","NS"] (DropEdge/NeuralSparse)
    "DE_p":0.2,
    "NS_k":5,
    "dropout":0.1
}