import torch
import torch.nn.functional as F
import numpy as np

def accuracy(output: torch.tensor, target: torch.tensor) -> float:
    with torch.no_grad():
        pred = torch.argmax(output, dim=1) # pred is the index of the maximal value in output => the predicted label of output
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3) -> float:
    """ Returns false positive rate"""
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def fpr(output: torch.tensor, target: torch.tensor) -> float:
    """ Returns false positive rate"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    fp = sum((pred == 1) & (target == 0))
    tn = sum((pred == 0) & (target == 0))
    fpr_val = fp / (fp + tn)
    if torch.isnan(fpr_val):
        return 0
    return fpr_val.item()

def fnr(output: torch.tensor, target: torch.tensor) -> float:
    """ Returns false negative rate"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    fn = sum((pred == 0) & (target == 1))
    tp = sum((pred == 1) & (target == 1))
    fnr_val = fn / (fn + tp)
    if torch.isnan(fnr_val):
        return 0
    return fnr_val.item()

def precision(output: torch.tensor, target: torch.tensor) -> float:
    """ Returns precision"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    tp = sum((pred == 1) & (target == 1))
    pp = sum((pred == 1))
    precision_val = tp / pp
    if torch.isnan(precision_val):
        return 0
    return precision_val.item()

def specificity(output: torch.tensor, target: torch.tensor) -> float:
    """ Returns specificity"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    fp = sum((pred == 1) & (target == 0))
    tn = sum((pred == 0) & (target == 0))
    specificity_val = tn /(fp+tn)
    if torch.isnan(specificity_val):
        return 0
    return specificity_val.item()

def recall(output: torch.tensor, target: torch.tensor) -> float:
    """ Returns recall"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    tp = sum((pred == 1) & (target == 1))
    fn = sum((pred == 0) & (target == 1))
    recall_val = tp /(tp+fn)
    if torch.isnan(recall_val):
        return 0
    return recall_val.item()

def mcc(output: torch.tensor, target: torch.tensor) -> float:
    """ Returns mathews correlation coefficient"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    fp = sum((pred == 1) & (target == 0))
    tp = sum((pred == 1) & (target == 1))
    fn = sum((pred == 0) & (target == 1))
    tn = sum((pred == 0) & (target == 0))
    mcc_val = (tp * tn - fp * fn) / torch.sqrt(((tp + fp) * (fn + tn) * (tp + fn) * (fp + tn)).float())
    if torch.isnan(mcc_val):
        return 0
    return mcc_val.item()

def f1_score(output: torch.tensor, target: torch.tensor) -> float:
    """ Returns mathews correlation coefficient"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    fp = sum((pred == 1) & (target == 0))
    tp = sum((pred == 1) & (target == 1))
    fn = sum((pred == 0) & (target == 1))
    f1_score_val = 2*tp/(2*tp +fp +fn)
    if torch.isnan(f1_score_val):
        return 0    
    return f1_score_val.item()
