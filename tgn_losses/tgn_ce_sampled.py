import torch
import torch.nn.functional as F
from torch import nn


class TGNCSE(nn.Module):
    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        logits = torch.cat((positive_scores.unsqueeze(1), negative_scores), dim=1)
        targets = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )
        return F.cross_entropy(logits, targets)
