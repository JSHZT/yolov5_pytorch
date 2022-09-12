from turtle import forward
import torch
import torch.nn as nn

def autopad(k):
    p = k // 2 if isinstance(k, int) else [i // 2 for i in k]
    return p

class CBL(nn.Module):
    def __init__(self, input_c, output_c, k_size=1, stride=1, pading=None, groups=1, act=True, e=1.0) -> None:
        super().__init__()
        input_c = round(input_c * e)
        output_c = round(output_c * e)
        self.conv = nn.Conv2d(input_c, output_c, k_size, stride, pading, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(output_c)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
class Focus(nn.Module):
    def __init__(self, input_c, output_c, k_size=1, stride=1, pading=None, groups=1, act=True, e=1.0) -> None:
        super().__init__()
        output_c = round(output_c * e)
        self.CBL = CBL(input_c*4, output_c, k_size, stride, pading, groups, act)
    
    def forward(self, x):
        flatten = torch.cat(
            x[..., 0::2, 0::2],
            x[..., 1::2, 0::2],
            x[..., 0::2, 1::2],
            x[..., 1::2, 1::2]
        )
        return self.CBL(flatten)
    
class SPP(nn.Module):
    def __init__(self,input_c, output_c, k_size=(5, 9, 13), e=1.0) -> None:
        super().__init__()
        input_c = round(input_c * e)
        output_c = round(output_c * e)
        c_ = input_c // 2
        self.CBL1 = CBL(input_c, c_)
        self.maxpool = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2) for k in k_size])
        self.CBL2 = CBL(c_ * 4, output_c)
        
    def forward(self, x):
        x = self.CBL1(x)
        return self.CBL2(torch.cat([x] + [m(x) for m in self.maxpool]))
    
class Res_unit_n(nn.Module):
    def __init__(self, input_c, output_c, n) -> None:
        super().__init__()
        self.shortout = input_c == output_c
        self.res_unit = nn.Sequential(
            CBL(input_c, input_c, 1, 1, 0),
            CBL(input_c, output_c, 3, 1, 1)
        )
        self.res_unit_n = nn.Sequential(*[self.res_unit for _ in range(n)])
        
    def forward(self, x):
        return x + self.res_unit_n(x) if self.shortout else self.res_unit_n(x)

class CSP1_x(nn.Module):
    def __init__(self, input_c, output_c, k_size=1, stride=1, pad=None, g=1, act=True, n=1, e=None) -> None:
        '''
        e:[depth, weight]深度和宽度控制参数
        '''
        super().__init__()        
        input_c = round(input_c * e[1])
        output_c = round(output_c * e[1])
        c_ = input_c // 2
        n = round(n * e[0])
        self.up = nn.Sequential(
            CBL(input_c, c_, k_size, stride, autopad(k_size), g, act),
            Res_unit_n(c_, c_, n),
            nn.Conv2d(c_, c_, 1, 1, 0, bias=False)
        )
        self.down = nn.Conv2d(input_c, c_, 1, 1, 0)
        self.tie = nn.Sequential(
            nn.BatchNorm2d(c_ * 2),
            nn.LeakyReLU(),
            CBL(c_ * 2, output_c, 1, 1, 0, bias=False)
        )
    
    def forward(self, x):
        return self.tie(torch.cat([self.up(x), self.down(x)], dim=1))
    
class CSP2_x(nn.Module):
    def __init__(self, input_c, output_c, k=1, s=1, p=None, g=1, act=True, n=1, e=None) -> None:
        '''
        e:[depth, weight]深度和宽度控制参数
        '''
        super().__init__()
        input_c = round(input_c * e[1])
        output_c = round(output_c * e[1])
        c_ = input_c // 2
        self.CBL_n = nn.Sequential(*[CBL(c_, c_, 1, 1, 0) for _ in range(n)])
        self.up = nn.Sequential(
            CBL(input_c, c_, k, s, autopad(k), g, act), 
            self.CBL_n,
            nn.Conv2d(c_, c_, 1, 1, 0)
        )
        self.down = nn.Conv2d(input_c, c_, 1, 1, 0, bias=False)
        self.tie = nn.Sequential(
            nn.BatchNorm2d(c_ * 2),
            nn.LeakyReLU(),
            CBL(c_ * 2, output_c, 1, 1, 0, bias=False)
        )
    
    def forward(self, x):
        return self.tie(torch.cat([self.up(x), self.down(x)], dim=1))