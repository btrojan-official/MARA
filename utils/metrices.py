import torch
from sklearn.metrics import roc_auc_score

def roc_auc(preds, labels):
    return roc_auc_score(labels.detach().cpu(), torch.softmax(preds.detach().cpu(), 1), multi_class='ovr')