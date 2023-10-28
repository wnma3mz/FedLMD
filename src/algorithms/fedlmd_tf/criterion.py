import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LMD_Tf_Loss"]


def lmd_tf_criterion(
    logits_student,
    target,
    T,
    major_labels,
):
    bs = logits_student.size(0)
    num_classes = logits_student.size(1)
    gt_mask = _get_gt_mask(logits_student, target)
    label_mask = torch.zeros_like(logits_student).scatter_(1, major_labels.repeat(bs, 1), 1).bool()

    # Build the target distribution
    pred_t = torch.ones((bs, num_classes)).to(logits_student.device) * 1. / (num_classes - len(label_mask) - 1)
    pred_t[label_mask] = 0
    pred_t[gt_mask] = 0.

    pred_s = F.log_softmax(logits_student / T - 1000 * gt_mask, dim=1)

    return nn.KLDivLoss(reduction="batchmean")(pred_s, pred_t) * (T ** 2)


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

class LMD_Tf_Loss(nn.Module):
    def __init__(self, num_classes=10, tau=1, beta=1):
        super(LMD_Tf_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets):
        ce_loss = self.CE(logits, targets)
        lmd_tf_loss = lmd_tf_criterion(logits, targets, self.tau)
        loss = ce_loss + self.beta * lmd_tf_loss
        return loss
