import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_Similarity



class Cross_inscl_loss(nn.Module):

    def __init__(self):
        super(Cross_inscl_loss, self).__init__()
    def forward(self, h0, h1):

        h0, h1 = nn.functional.normalize(h0, dim=1), nn.functional.normalize(h1, dim=1)
        cos12 = get_Similarity(h0, h1)
        cos11 = get_Similarity(h0, h0)
        cos22 = get_Similarity(h1, h1)
        cos21 = get_Similarity(h1, h0)
        sim12 = (cos12).exp()
        sim11 = (cos11).exp()
        sim21 = (cos21).exp()
        sim22 = (cos22).exp()
        pos1 = sim12.diag()
        pos2 = sim21.diag()
        p1 = pos1 / (sim12 + sim11).sum(1)
        p2 = pos2 / (sim21 + sim22).sum(1)
        loss1 = (p1 * (torch.log(p1 / p2))).sum()
        loss2 = (p2 * (torch.log(p2 / p1))).sum()

        loss = (loss1+loss2)/2
        return loss
class Noise_robust_loss(nn.Module):
    def __init__(self):
        super(Noise_robust_loss, self).__init__()
    def forward(self,h0, h1):
        h0, h1 = nn.functional.normalize(h0, dim=1), nn.functional.normalize(h1, dim=1)
        tao = 1
        t = 3
        cos = get_Similarity(h0, h1)
        sim = (cos/tao).exp()
        pos = sim.diag()
        Q = pos / sim.sum(1)
        result = 0.0000
        for i in range(1, t):

            result += (((1 - Q) ** i) / i).mean()
        robust_loss = result
        return robust_loss




