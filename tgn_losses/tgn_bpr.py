import torch
import torch.nn.functional as F
from torch import nn


class TGNBPR(nn.Module):

    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        score_differences = positive_scores.unsqueeze(1) - negative_scores
        return -F.logsigmoid(score_differences).mean()
