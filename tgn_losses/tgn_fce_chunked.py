import torch
import torch.nn.functional as F
from torch import nn


class TGNFCE(nn.Module):
    def forward(self, catalog_scores: torch.Tensor, positive_item_ids: torch.Tensor):
        return F.cross_entropy(catalog_scores, positive_item_ids.long())
