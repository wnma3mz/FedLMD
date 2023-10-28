# FedLMD (Federated Learning with Label-Masking Distillation)

## Run

For easy reproduction, we use [FedNTD](https://github.com/wnma3mz/FedLMD/tree/master/src#how-to-run-codes). 

```bash
cd src/ && python ./main.py --config_path ./config/${algorithm_name}.json
```

## Key Code

```python

def lmd_criterion(
    logits_s,
    logits_t,
    target,
    T,
    major_labels,
    use_teacher=True
):
    bs = logits_s.size(0)
    num_classes = logits_student.size(1)
    gt_mask = _get_gt_mask(logits_s, target)
    label_mask = torch.zeros_like(logits_s).scatter_(1, major_labels.repeat(bs, 1), 1).bool()

    pred_s = F.log_softmax(logits_s / T - 1000 * gt_mask, dim=1)
    if use_teacher:
        pred_t = F.softmax(logits_t / T - 1000 * label_mask - 1000 * gt_mask, dim=1)
    else:
        # Build the target distribution. Don't need the teacher logits.
        pred_t = torch.ones((bs, num_classes)).to(logits_s.device) * 1. / (num_classes - len(label_mask) - 1)
        pred_t[label_mask] = 0
        pred_t[gt_mask] = 0.

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

    def forward(self, logits, targets, dg_logits, local_data_label, use_teacher):
        ce_loss = self.CE(logits, targets)
        lmd_loss = lmd_criterion(logits, dg_logits, targets, self.tau, local_data_label)
        loss = ce_loss + self.beta * lmd_loss
        return loss
```

## Reference

We using to the following repositories:

https://github.com/Lee-Gihun/FedNTD

## Citing this work

```bibtex
@inproceedings{10.1145/3581783.3611984,
    author = {Lu, Jianghu and Li, Shikun and Bao, Kexin and Wang, Pengju and Qian, Zhenxing and Ge, Shiming},
    title = {Federated Learning with Label-Masking Distillation},
    year = {2023},
    isbn = {9798400701085},
    booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
    pages = {222–232},
    numpages = {11},
}
```

