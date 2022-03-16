import torch 
from torch import nn


class KDLoss(nn.Module):
    def __init__(self, T: float, alpha: float) -> None:
        super(KDLoss, self).__init__()
        self.T = T  # temperature scale
        self.alpha = alpha

    def forward(
        self,
        student_preds: torch.Tensor,  # num_samples x num_classes
        teacher_preds: torch.Tensor,  # num_samples x num_classes
        targets: torch.Tensor  # num_samples
    ) -> float:
        # KL Divergence Loss for the first component of KD loss
        loss1 = nn.KLDivLoss()(
            nn.LogSoftmax(dim=1)(student_preds / self.T),
            nn.Softmax(dim=1)(teacher_preds / self.T)
        )

        # CrossEntropy Loss for the second component of KD loss
        loss2 = nn.CrossEntropyLoss()(student_preds, targets)

        # Combination
        KD_loss = self.alpha * (self.T ** 2) * loss1 + (1 - self.alpha) * loss2

        return KD_loss
