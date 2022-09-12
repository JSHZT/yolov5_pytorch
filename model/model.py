import torch
import torch.nn as nn
from modules import *

class CSPDarkNet(nn.Module):
    def __init__(self, gd=0.33, gw=0.5):
        super(CSPDarkNet, self).__init__()
        self.truck_big = nn.Sequential(
            Focus(3, 64, e=gw),
            CBL(64, 128, 3, 2, 1, e=gw),
            CSP1_x(128, 128, n=3, e=[gd, gw]),
            CBL(128, 256, 3, 2, 1, e=gw),
            CSP1_x(256, 256, n=9, e=[gd, gw]),
        )
        self.truck_middle = nn.Sequential(
            CBL(256, 512, k=3, s=2, p=1, e=gw),
            CSP1_x(512, 512, n=9, e=[gd, gw]),
        )
        self.truck_small = nn.Sequential(
            CBL(512, 1024, k=3, s=2, p=1, e=gw),
            SPP(1024, 1024, e=gw)
        )
        
    def forward(self, x):
        h_big = self.truck_big(x)  # torch.Size([2, 128, 76, 76])
        h_middle = self.truck_middle(h_big)
        h_small = self.truck_small(h_middle)
        return h_big, h_middle, h_small

def darknet53(gd, gw, pretrained, **kwargs):
    model = CSPDarkNet(gd, gw)
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception(f"darknet request a pretrained path. got[{pretrained}]")
    return model


class yolov5(nn.modules):
    def __init__(self) -> None:
        super().__init__()