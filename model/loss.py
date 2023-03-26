import torch.nn as nn
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target) -> float:
    # this function already do log_softmax to output and then do nll_loss
    # equivalent to nll_loss(F.log_softmax(output, dim=1), target)
    return F.cross_entropy(output, target)
