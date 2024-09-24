import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.losses import DiceLoss
import warnings
import numpy as np
import sys

class CompoundLoss(nn.Module):
    def __init__(self, loss1, loss2=None, alpha1=1., alpha2=0.):
        super(CompoundLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, y_pred, y_true):
        l1 = self.loss1(y_pred, y_true)
        if self.alpha2 == 0 or self.loss2 is None:
            return self.alpha1*l1
        l2 = self.loss2(y_pred, y_true)
        return self.alpha1*l1 + self.alpha2 * l2

# just a wrapper to make sure you are not using softmax before the loss computation
class DSCLoss(nn.Module):
    def __init__(self, include_background=False):
        super(DSCLoss, self).__init__()
        self.loss = DiceLoss(softmax=True, include_background=include_background)
        self.check_softmax = True

    def forward(self, y_pred, y_true):
        if self.check_softmax:
            if y_pred.softmax(dim=1).flatten(2).sum(dim=1).mean(dim=1).mean() != 1.0:
                # flatten(2) flattens all after dim=2, sum over classes, take mean to get per-batch values, & take mean
                warnings.warn('check you did not apply softmax before loss computation')
            else: self.check_softmax = False
        return self.loss(y_pred, y_true)

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


def get_loss(loss1, loss2=None, alpha1=1., alpha2=0.):
    if loss1 == loss2:
        warnings.warn('using same loss twice, you sure?')
    loss_dict = dict()
    loss_dict['ce'] = CELoss()
    loss_dict['dice'] = DSCLoss()
    loss_dict['cedice'] = CompoundLoss(CELoss(), DSCLoss(), alpha1=1., alpha2=1.)
    loss_dict[None] = None

    loss_fn = CompoundLoss(loss_dict[loss1], loss_dict[loss2], alpha1, alpha2)

    return loss_fn



