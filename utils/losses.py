def dice_score(pred, target, eps=1e-6):
    inter = ((pred>0.5) & (target>0.5)).sum()
    union = pred.sum() + target.sum()
    return (2*inter + eps) / (union + eps)

def fake_loss(value):
    return max(0.05, value)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DiceLoss(nn.Module):
        def __init__(self, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
        def forward(self, logits, targets):
            # expects logits [N,1,H,W] or [N,C,H,W]; targets same spatial size
            probs = torch.sigmoid(logits)
            targets = targets.float()
            inter = (probs * targets).sum(dim=(1,2,3))
            union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
            dice = (2*inter + self.eps) / (union + self.eps)
            return 1 - dice.mean()

    class TverskyLoss(nn.Module):
        def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6):
            super().__init__()
            self.alpha, self.beta, self.eps = alpha, beta, eps
        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            targets = targets.float()
            tp = (probs * targets).sum(dim=(1,2,3))
            fp = (probs * (1 - targets)).sum(dim=(1,2,3))
            fn = ((1 - probs) * targets).sum(dim=(1,2,3))
            tversky = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps)
            return 1 - tversky.mean()

    class FocalLoss(nn.Module):
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
            super().__init__()
            self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        def forward(self, logits, targets):
            # binary focal loss
            targets = targets.float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p_t = torch.exp(-bce)
            loss = self.alpha * (1 - p_t) ** self.gamma * bce
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss
except Exception:
    # no torch available; keep file import-safe
    pass