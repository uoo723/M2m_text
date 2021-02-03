"""
Created on 2021/01/07
@author Sangwoo Han
"""
import torch


def classwise_loss(outputs, targets, multi_label=False):
    """
    Reference: https://github.com/alinlab/M2m/blob/master/utils.py
    """
    if multi_label:
        return (outputs * targets).mean()

    out_1hot = torch.ones_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), -1)
    return (outputs * out_1hot).mean()
