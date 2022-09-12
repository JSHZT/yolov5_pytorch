import torch
import torch.nn as nn
from modules import *

class backbone(nn.Module):
    def __init__(self, gd=0.33, gw=0.5):
        super(backbone, self).__init__()
        self.truck_big = nn.Sequential(
            Focus(3, 64, e=gw),
            CBL(64, 128, k_size=3, stride=2, pading=1, e=gw),#降采样
            CSP1_x(128, 128, n=3, e=[gd, gw]),
            CBL(128, 256, k_size=3, stride=2, pading=1, e=gw),#降采样
            CSP1_x(256, 256, n=9, e=[gd, gw]),
        )
        self.truck_middle = nn.Sequential(
            CBL(256, 512, k_size=3, stride=2, pading=1, e=gw), #降采样
            CSP1_x(512, 512, n=9, e=[gd, gw]),
        )
        self.truck_small = nn.Sequential(
            CBL(512, 1024, k_size=3, stride=2, pading=1, e=gw),#降采样
            SPP(1024, 1024, e=gw),
        )
        
    def forward(self, x):
        h_big = self.truck_big(x)  # torch.Size([2, 128, 76, 76])
        h_middle = self.truck_middle(h_big)
        h_small = self.truck_small(h_middle)
        return h_big, h_middle, h_small

def get_backbone(gd, gw, pretrained, **kwargs):
    model = backbone(gd, gw)
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception(f"darknet request a pretrained path. got[{pretrained}]")
    return model


class yolov5(nn.modules):
    def __init__(self, nc=80, gd=0.33, gw=0.5):
        super(yolov5, self).__init__()
        self.backbone = get_backbone(gd, gw, None)

        self.neck_small = nn.Sequential(
            CSP1_x(1024, 1024, n=3, e=[gd, gw]),
            CBL(1024, 512, 1, 1, 0, e=gw)
        )
        self.up_middle = nn.Sequential(
            UPsample()
        )
        self.out_set_middle = nn.Sequential(
            CSP1_x(1024, 512, n=3, e=[gd, gw]),
            CBL(512, 256, 1, 1, 0, e=gw),
        )
        self.up_big = nn.Sequential(
            UPsample()
        )
        self.out_set_tie_big = nn.Sequential(
            CSP1_x(512, 256, n=3, e=[gd, gw])
        )

        self.pan_middle = nn.Sequential(
            CBL(256, 256, 3, 2, 1, e=gw)
        )
        self.out_set_tie_middle = nn.Sequential(
            CSP1_x(512, 512, n=3, e=[gd, gw])
        )
        self.pan_small = nn.Sequential(
            CBL(512, 512, 3, 2, 1, e=gw)
        )
        self.out_set_tie_small = nn.Sequential(
            CSP1_x(1024, 1024, n=3, e=[gd, gw])
        )
        big_ = round(256 * gw)
        middle = round(512 * gw)
        small_ = round(1024 * gw)
        self.out_big = nn.Sequential(
            nn.Conv2d(big_, 3 * (5 + nc), 1, 1, 0)
        )
        self.out_middle = nn.Sequential(
            nn.Conv2d(middle, 3 * (5 + nc), 1, 1, 0)
        )
        self.out_small = nn.Sequential(
            nn.Conv2d(small_, 3 * (5 + nc), 1, 1, 0)
        )

    def forward(self, x):
        h_big, h_middle, h_small = self.backbone(x)
        neck_small = self.neck_small(h_small)  
        up_middle = self.up_middle(neck_small)
        middle_cat = torch.cat([up_middle, h_middle], dim=1)
        out_set_middle = self.out_set_middle(middle_cat)

        up_big = self.up_big(out_set_middle)  # torch.Size([2, 128, 76, 76])
        big_cat = torch.cat([up_big, h_big], dim=1)
        out_set_tie_big = self.out_set_tie_big(big_cat)

        neck_tie_middle = torch.cat([self.pan_middle(out_set_tie_big), out_set_middle], dim=1)
        up_middle = self.out_set_tie_middle(neck_tie_middle)

        neck_tie_small = torch.cat([self.pan_small(up_middle), neck_small], dim=1)
        out_set_small = self.out_set_tie_small(neck_tie_small)

        out_small = self.out_small(out_set_small)
        out_middle = self.out_middle(up_middle)
        out_big = self.out_big(out_set_tie_big)

        return out_small, out_middle, out_big


if __name__ == '__main__':
    config = {
        #            gd    gw
        'yolov5s': [0.33, 0.50],
        'yolov5m': [0.67, 0.75],
        'yolov5l': [1.00, 1.00],
        'yolov5x': [1.33, 1.25]
    }
    net_size = config['yolov5x']
    net = yolov5(nc=80, gd=net_size[0], gw=net_size[1])
    print(net)
    a = torch.randn(2, 3, 416, 416)
    y = net(a)
    print(y[0].shape, y[1].shape, y[2].shape)
 