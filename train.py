import torch
from torch.utils.tensorboard import SummaryWriter

from utils.metrices import roc_auc
from utils.read_data import IMDB_mlh
from utils.graph_info import graph_info
from MARA.MARA import MARA

from config import config

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Working on device: " + device + "\n")

imdb = IMDB_mlh().to(device)
imdb.info()

train_mask, val_mask, test_mask = imdb.get_training_mask(train_mask_size=0.25, val_mask_size=0.25)

def train(data, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    intra_layer_edges = [data.layer_1, data.layer_2]
    cross_layer_edges = [data.cross_edges]

    out, _, _ = model(data.node_features, data.node_layers, intra_layer_edges, cross_layer_edges)

    train_loss = criterion(out[train_mask], data.classes[train_mask])
    train_loss.backward()
    optimizer.step()

    train_score = roc_auc(out[train_mask], data.classes[train_mask])
    val_score = roc_auc(out[val_mask], data.classes[val_mask])

    return train_loss.item(), train_score, val_score

def evaluate(data, model, mask):
    model.eval()
    with torch.no_grad():
        graph_info(data.classes, data.node_layers, [data.layer_1, data.layer_2], [data.cross_edges])
        
        intra_layer_edges = [data.layer_1, data.layer_2]
        cross_layer_edges = [data.cross_edges]

        out, intra_layer_edges, cross_layer_edges = model(data.node_features, data.node_layers, intra_layer_edges, cross_layer_edges)

        graph_info(data.classes, data.node_layers, intra_layer_edges, cross_layer_edges)

        score = roc_auc(out[mask], data.classes[mask])
    return score


writer = SummaryWriter()

# If you want to run this without early stopping, set patience to 
# bigger number than epoch number

# Patience is based on vaidation roc_auc, not loss!

early_stopping = {
    "best_val_score": 0,
    "patience": config["patience"],
    "counter": 0,
    "best_weights": None
}

mara = MARA().to(device)
crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(mara.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

print("STARTING TRAINING:")
for epoch in range(config["epoch_num"]):
    train_loss, train_score, val_score = train(imdb, mara, crit, optim)

    if val_score > early_stopping["best_val_score"]:
        early_stopping = {"best_val_score": val_score, "counter": 0, "best_weights": mara.state_dict(), "patience":early_stopping["patience"]}
    else:
        early_stopping["counter"] += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Train AUC: {train_score:.4f} | Val AUC: {val_score:.4f}")

    if early_stopping["counter"] >= early_stopping["patience"]:
        print(f"Early stopping at epoch {epoch+1}")
        break

    writer.add_hparams(
        config,
        {"val_score": val_score, "train_score": train_score, "loss": train_loss, "epoch": epoch+1},
    )

mara.load_state_dict(early_stopping["best_weights"])

print("===== TEST: =====")
test_score = evaluate(imdb, mara, test_mask)
print("===== WHOLE: =====")
whole_score = evaluate(imdb, mara, slice(None))
print(f"Final Test AUC: {test_score:.4f} | Whole Dataset AUC: {whole_score:.4f}")

writer.close()
