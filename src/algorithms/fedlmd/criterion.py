import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LMD_Loss"]


def lmd_criterion(
    logits_student,
    logits_teacher,
    target,
    T,
    major_labels,
):
    bs = logits_student.size(0)
    gt_mask = _get_gt_mask(logits_student, target)
    label_mask = torch.zeros_like(logits_student).scatter_(1, major_labels.repeat(bs, 1), 1).bool()

    pred_t = F.softmax(logits_teacher / T - 1000 * label_mask - 1000 * gt_mask, dim=1)
    pred_s = F.log_softmax(logits_student / T - 1000 * gt_mask, dim=1)

    return nn.KLDivLoss(reduction="batchmean")(pred_s, pred_t) * (T ** 2)


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

class LMD_Loss(nn.Module):
    def __init__(self, num_classes=10, tau=1, beta=1):
        super(LMD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits, major_labels):
        ce_loss = self.CE(logits, targets)
        lmd_loss = lmd_criterion(logits, dg_logits, targets, self.tau, major_labels)
        loss = ce_loss + self.beta * lmd_loss
        return loss
