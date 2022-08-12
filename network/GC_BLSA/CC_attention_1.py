'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention_1(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention_1,self).__init__()
        self.query_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF_0 = INF

        self.gamma = nn.Parameter(torch.zeros(1))

        self.query_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv_1 = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8, kernel_size=1)
        self.value_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.INF_1 = INF

    def forward(self, x_0, x_1):
        m_batchsize, _, height, width = x_0.size()  # （B, C, H, W）

        proj_query_0 = self.query_conv_0(x_0)   # （B, C, H, W）=（B, 64, 97, 97）
        proj_query_H_0 = proj_query_0.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)  # （B*W, H, C）
        proj_query_W_0 = proj_query_0.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)  # （B*H, W, C）
        proj_key_0 = self.key_conv_0(x_0)  # （B, 64, H, W）
        proj_key_H_0 = proj_key_0.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)   # （B*W, C, H）
        proj_key_W_0 = proj_key_0.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)   # （B*H, C, W)
        proj_value_0 = self.value_conv_0(x_0)   # (B, 512, H, W)
        proj_value_H_0 = proj_value_0.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # (B*W, C, H)
        proj_value_W_0 = proj_value_0.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # (B*H, C, W)

        proj_query_1 = self.query_conv_1(x_1)
        proj_query_H_1 = proj_query_1.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W_1 = proj_query_1.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key_1 = self.key_conv_1(x_1)
        proj_key_H_1 = proj_key_1.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W_1 = proj_key_1.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value_1 = self.value_conv_1(x_1)
        proj_value_H_1 = proj_value_1.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W_1 = proj_value_1.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        energy_H_0 = (torch.bmm(proj_query_H_1, proj_key_H_0)+self.INF_0(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W_0 = torch.bmm(proj_query_W_1, proj_key_W_0).view(m_batchsize,height,width,width)
        concate_0 = self.softmax(torch.cat([energy_H_0, energy_W_0], 3))

        energy_H_1 = (torch.bmm(proj_query_H_0, proj_key_H_1) + self.INF_1(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W_1 = torch.bmm(proj_query_W_0, proj_key_W_1).view(m_batchsize, height, width, width)
        concate_1 = self.softmax(torch.cat([energy_H_1, energy_W_1], 3))

        att_H_0 = concate_0[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W_0 = concate_0[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H_0 = torch.bmm(proj_value_H_0, att_H_0.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W_0 = torch.bmm(proj_value_W_0, att_W_0.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        cat_0 = self.gamma*(out_H_0 + out_W_0) + x_0

        att_H_1 = concate_1[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W_1 = concate_1[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H_1 = torch.bmm(proj_value_H_1, att_H_1.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W_1 = torch.bmm(proj_value_W_1, att_W_1.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        cat_1 = self.gamma * (out_H_1 + out_W_1) + x_1

        return cat_0, cat_1



# if __name__ == '__main__':
#     model = CrissCrossAttention_1(64)
#     x = torch.randn(2, 64, 5, 6)
#     out = model(x)
#     print(out.shape)
