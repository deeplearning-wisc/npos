import torch.nn as nn
import torch
import torch.nn.functional as F

class LogitNormLoss(nn.Module):

    def __init__(self, device='cuda', t=8.):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)