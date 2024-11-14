config = {
    "simplification_type":"l-b-l", # ["l-b-l","multilayer"]
    "simplification_stages":"each", # ["once","each"]
    "simplification_strategy":"DE", # ["DE","NS"] (DropEdge/NeuralSparse)
    "DE_p":0.2,
    "NS_k":5,
}