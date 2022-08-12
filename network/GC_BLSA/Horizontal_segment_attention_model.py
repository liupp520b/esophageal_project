import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

class KQVConv(nn.Module):
    def __init__(self, in_dim):
        super(KQVConv, self).__init__()
        self.conv_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.conv_q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):
        K = self.conv_k(x)
        Q = self.conv_q(x)
        V = self.conv_v(x)
        return K, Q, V

class selfAttention(nn.Module):
    def __init__(self, poolscale, name='HSA'):
        super(selfAttention, self).__init__()
        self.scale = poolscale
        # self.space_pool_K = nn.Conv2d(in_channels=dim_k, out_channels=dim_k, kernel_size=(self.scale, self.scale), stride=(4, 4), bias=True)
        # self.space_pool_V = nn.Conv2d(in_channels=dim_v, out_channels=dim_v, kernel_size=(self.scale, self.scale), stride=(4, 4), bias=True)
        if name == 'HSA':
            self.space_pool_K = nn.AvgPool2d(kernel_size=(self.scale, self.scale*4), stride=(self.scale, self.scale*4))
            self.space_pool_V = nn.AvgPool2d(kernel_size=(self.scale, self.scale * 4), stride=(self.scale, self.scale * 4))
        if name == 'VSA':
            self.space_pool_K = nn.AvgPool2d(kernel_size=(self.scale * 4, self.scale), stride=(self.scale, self.scale * 4))
            self.space_pool_V = nn.AvgPool2d(kernel_size=(self.scale * 4, self.scale),stride=(self.scale, self.scale * 4))

    def forward(self, K, Q, V):
        Q_batchsize, Q_chanel, Q_heigth, Q_width = Q.size()
        Q_pie = Q.contiguous().view(Q_batchsize, Q_chanel, Q_heigth*Q_width).permute(0, 2, 1)   # (B, HW, C)
        # print("Q_pie:",Q_pie.size())
        K = self.space_pool_K(K)
        K_batchsize, K_chanel, K_heigth, K_width = K.size()
        K_pie = K.contiguous().view(K_batchsize, -1, K_heigth*K_width)        # (B, C, H_jian*W_jian)
        V = self.space_pool_V(V)
        V_batchsize, V_chanel, V_heigth, V_width = V.size()
        V_pie = V.contiguous().view(V_batchsize, -1, V_heigth*V_width)  # (B, C, H_pie*W_pie)
        attention = F.softmax(torch.bmm(Q_pie, K_pie), dim=1)  # (B, HW, H_pie*W_pie)
        # attention_batchsize, attention_heigth, attention_width = attention.size()
        attention = attention.permute(0, 2, 1)
        # print("attention:",attention.size())
        out = torch.bmm(V_pie, attention)
        # print("out:",out.size())
        out = out.view(Q_batchsize, -1, Q_heigth, Q_width)
        return  out

class HSA_model(nn.Module):
    """Horizontal Segment Attention Module"""
    def __init__(self, in_dim):
        super(HSA_model, self).__init__()
        self.KQV_0 = KQVConv(in_dim)
        self.KQV_1 = KQVConv(in_dim)
        self.KQV_2 = KQVConv(in_dim)
        self.KQV_3 = KQVConv(in_dim)

        self.self_att_0 = selfAttention(poolscale=1, name='HSA')   # 16*16
        self.self_att_1 = selfAttention(poolscale=2, name='HSA')   # 8*8
        self.self_att_2 = selfAttention(poolscale=4, name='HSA')   # 4*4
        self.self_att_3 = selfAttention(poolscale=8, name='HSA')   # 2*2

        # self.outconv = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=5, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(5),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        batchsize, chanel, height, width = x.size()
        # print(x.size())
        # print("batchsize:{}, chanel:{}, height:{}, width:{}".format(batchsize, chanel, height, width))
        block_0 = x[:,:,0:(height//4),:]
        # print("block_0:",block_0.size())
        block_1 = x[:,:,(height//4):(height//2),:]
        # print("block_1:", block_1.size())
        block_2 = x[:,:,height//2:(3*height//4),:]
        block_3 = x[:,:,(3*height//4):(height),:]    # (4, 2048, 16, 64)
        K_0, Q_0, V_0 = self.KQV_0(block_0)   # K_0 (4, 256, 16, 64)   Q_0 (4, 256, 16, 64)   V_0  (4, 2048, 16, 64)
        K_1, Q_1, V_1 = self.KQV_1(block_1)
        K_2, Q_2, V_2 = self.KQV_2(block_2)
        K_3, Q_3, V_3 = self.KQV_3(block_3)
        # print("K_0:",K_0.size())
        # print("Q_0:", Q_0.size())
        # print("V_0:", V_0.size())

        out_0 = self.self_att_0(K_0, Q_1, V_0)
        out_1 = self.self_att_1(K_1, Q_2, V_1)
        out_2 = self.self_att_2(K_2, Q_3, V_2)
        out_3 = self.self_att_3(K_3, Q_0, V_3)

        out = torch.cat([out_0, out_1, out_2, out_3], dim=2)
        # print("out",out.size())
        # out = self.outconv(out)
        return out

class VSA_model(nn.Module):
    """Horizontal Segment Attention Module"""
    def __init__(self, in_dim):
        super(VSA_model, self).__init__()
        self.KQV_0 = KQVConv(in_dim)
        self.KQV_1 = KQVConv(in_dim)
        self.KQV_2 = KQVConv(in_dim)
        self.KQV_3 = KQVConv(in_dim)

        self.self_att_0 = selfAttention(poolscale=1, name='VSA')  # 16*16
        self.self_att_1 = selfAttention(poolscale=2, name='VSA')  # 8*8
        self.self_att_2 = selfAttention(poolscale=4, name='VSA')  # 4*4
        self.self_att_3 = selfAttention(poolscale=8, name='VSA')  # 2*2

        # self.outconv = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=512, out_channels=5, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(5),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        batchsize, chanel, height, width = x.size()
        block_0 = x[:, :, :, 0:(width//4)]
        block_1 = x[:, :, :, (width//4):(width//2)]
        block_2 = x[:, :, :, (width//2):(3*width//4)]
        block_3 = x[:, :, :, (3*width//4):width]

        K_0, Q_0, V_0 = self.KQV_0(block_0)  # K_0 (4, 256, 64, 16)   Q_0 (4, 256, 64, 16)   V_0  (4, 2048, 64, 16)
        K_1, Q_1, V_1 = self.KQV_1(block_1)
        K_2, Q_2, V_2 = self.KQV_2(block_2)
        K_3, Q_3, V_3 = self.KQV_3(block_3)

        out_0 = self.self_att_0(K_0, Q_1, V_0)
        out_1 = self.self_att_1(K_1, Q_2, V_1)
        out_2 = self.self_att_2(K_2, Q_3, V_2)
        out_3 = self.self_att_3(K_3, Q_0, V_3)

        out = torch.cat([out_0, out_1, out_2, out_3], dim=3)
        # out = self.outconv(out)

        return  out