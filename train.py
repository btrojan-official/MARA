import torch
from torch.utils.tensorboard import SummaryWriter

from utils.metrices import roc_auc
from utils.read_data import IMDB_mlh
from MARA.MARA import MARA

from config import config

params = {
    "simplification_type":config["simplification_type"],
    "simplification_stages":config["simplification_stages"],
    "DE_p":config["DE_p"],
    "lr":2e-3,
    "weight_decay":0.0005,
    "dropout":config["dropout"]
    }

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Working on device: ", device)

imdb = IMDB_mlh().to(device)
imdb.info()

train_mask, val_mask, test_mask = imdb.get_training_mask(train_mask_size=0.25, val_mask_size=0.25)

def train(data, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    edges = torch.cat([data.layer_1, data.layer_2, data.cross_edges], dim=0).t()
    layers_lengths = torch.tensor([data.layer_1.shape[0], data.layer_2.shape[0], data.cross_edges.shape[0]], dtype=torch.int64)

    out, h = model(data.node_features, edges, layers_lengths)

    train_loss = criterion(out[train_mask], data.classes[train_mask])
    train_loss.backward()
    optimizer.step()

    train_score = roc_auc(out[train_mask], data.classes[train_mask])
    val_score = roc_auc(out[val_mask], data.classes[val_mask])

    return train_loss.item(), train_score, val_score

def evaluate(data, model, mask):
    model.eval()
    with torch.no_grad():
        edges = torch.cat([data.layer_1, data.layer_2, data.cross_edges], dim=0).t()
        layers_lengths = torch.tensor([data.layer_1.shape[0], data.layer_2.shape[0], data.cross_edges.shape[0]], dtype=torch.int64)

        out, h = model(data.node_features, edges, layers_lengths)

        score = roc_auc(out[mask], data.classes[mask])
    return score


writer = SummaryWriter()

# If you want to run this without early stopping, set patience to 
# bigger number than epoch number

# Patience is based on vaidation roc_auc, not loss!

early_stopping = {
    "best_val_score": 0,
    "patience": 50,
    "counter": 0,
    "best_weights": None
}

mara = MARA(simplification_type=params["simplification_type"], simplification_stages=params["simplification_stages"], DE_p=params["DE_p"], dropout=params["dropout"]).to(device)
crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(mara.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

print("STARTING TRAINING:")
for epoch in range(251):
    train_loss, train_score, val_score = train(imdb, mara, crit, optim)

    if val_score > early_stopping["best_val_score"]:
        early_stopping = {"best_val_score": val_score, "counter": 0, "best_weights": mara.state_dict(), "patience":early_stopping["patience"]}
    else:
        early_stopping["counter"] += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Train AUC: {train_score:.4f} | Val AUC: {val_score:.4f}")

    if early_stopping["counter"] >= early_stopping["patiWence"]:
        print(f"Early stopping at epoch {epoch+1}")
        break

    writer.add_hparams(
        params,
        {"val_score": val_score, "train_score": train_score, "loss": train_loss, "epoch": epoch+1},
    )

mara.load_state_dict(early_stopping["best_weights"])

test_score = evaluate(imdb, mara, test_mask)
whole_score = evaluate(imdb, mara, slice(None))
print(f"Final Test AUC: {test_score:.4f} | Whole Dataset AUC: {whole_score:.4f}")

writer.close()
