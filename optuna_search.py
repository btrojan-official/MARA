import torch
import optuna
import optuna.visualization as vis
from torch.utils.tensorboard import SummaryWriter

from utils.metrices import roc_auc
from utils.read_data import IMDB_mlh
from MARA.MARA import MARA

from config import config

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Working on device: ", device)

imdb = IMDB_mlh().to(device)
imdb.info()

train_mask, val_mask, test_mask = imdb.get_training_mask(train_mask_size=0.25, val_mask_size=0.25)

writer = SummaryWriter(log_dir="./tensorboard_logs")

def train(data, model, criterion, optimizer, epoch, trial_number):
    model.train()
    optimizer.zero_grad()

    edges = torch.cat([data.layer_1, data.layer_2, data.cross_edges], dim=0).t()
    layers_lengths = torch.tensor([data.layer_1.shape[0], data.layer_2.shape[0], data.cross_edges.shape[0]], dtype=torch.int64)

    out, _, _ = model(data.node_features, edges, layers_lengths)

    train_loss = criterion(out[train_mask], data.classes[train_mask])
    train_loss.backward()
    optimizer.step()

    train_score = roc_auc(out[train_mask], data.classes[train_mask])
    val_score = roc_auc(out[val_mask], data.classes[val_mask])

    writer.add_scalar(f"Trial {trial_number}/Loss/train", train_loss.item(), epoch)
    writer.add_scalar(f"Trial {trial_number}/ROC-AUC/train", train_score, epoch)
    writer.add_scalar(f"Trial {trial_number}/ROC-AUC/val", val_score, epoch)

    return train_loss.item(), train_score, val_score

def objective(trial):
    params = {
        "simplification_type": trial.suggest_categorical("simplification_type", ["multilayer", "l-b-l"]),
        "simplification_stages": trial.suggest_categorical("simplification_stages", ["once", "each"]),
        "DE_p": trial.suggest_float("DE_p", 0, 0.7),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0, 0.7)
    }

    mara = MARA(
        simplification_type=params["simplification_type"],
        simplification_stages=params["simplification_stages"],
        DE_p=params["DE_p"],
        dropout=params["dropout"]
    ).to(device)
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(mara.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    early_stopping = {
        "best_val_score": 0,
        "patience": 15,
        "counter": 0,
        "best_weights": None
    }

    for epoch in range(251):
        train_loss, train_score, val_score = train(imdb, mara, crit, optim, epoch, trial.number)

        if val_score > early_stopping["best_val_score"]:
            early_stopping = {
                "best_val_score": val_score,
                "counter": 0,
                "best_weights": mara.state_dict(),
                "patience": early_stopping["patience"]
            }
        else:
            early_stopping["counter"] += 1

        if early_stopping["counter"] >= early_stopping["patience"]:
            break

    mara.load_state_dict(early_stopping["best_weights"])

    trial_number = trial.number
    writer.add_scalar(f"Trial {trial_number}/Best ROC-AUC/val", early_stopping["best_val_score"], trial_number)
    for param_name, param_value in params.items():
        writer.add_text(f"Trial {trial_number}/Params", f"{param_name}: {param_value}", trial_number)

    return early_stopping["best_val_score"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

vis.plot_optimization_history(study)
vis.plot_param_importances(study)
vis.plot_slice(study)

best_trial = study.best_trial
writer.add_text("Best Trial/Value", f"Best ROC-AUC: {best_trial.value}")
for param_name, param_value in best_trial.params.items():
    writer.add_text(f"Best Trial/Params", f"{param_name}: {param_value}")

for trial in study.trials:
    trial_number = trial.number
    writer.add_scalar(f"All Trials/Best ROC-AUC", trial.value, trial_number)
    for param_name, param_value in trial.params.items():
        writer.add_text(f"All Trials/Params {trial_number}", f"{param_name}: {param_value}")

print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

writer.close()
