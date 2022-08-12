import torch
from torch import nn
from  torch.nn import functional as F

class block(nn.Module):
    def __init__(self, in_channel):
        super(block, self).__init__()

        self.in_channel = in_channel
        self.inter_channel = in_channel // 2
        self.g = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1,
                           stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1,
                           stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1,
                               stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1,
                    stride=1, padding=0),
            nn.BatchNorm2d(self.in_channel)
        )
    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.in_channel, -1)
        g_x = g_x.permute(0, 2 ,1)

        theta_x = self.theta(x).view(batch_size, self.in_channel, -1)
        theta_x = theta_x.permute(0, 2 ,1)

        phi_x = self.phi(x).view(batch_size, self.in_channel, -1)
        f = torch.matmul(theta_x, phi_x)
        f_dive_C = F.softmax(f, dim=-1)
        # print(f_dive_C.size())
        y = torch.matmul(f_dive_C, g_x)
        y = y.permute(0, 2 ,1).contiguous()
        y = y.view(batch_size, self.in_channel, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z