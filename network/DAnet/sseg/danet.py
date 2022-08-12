###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from torch.nn import functional as F
from torch.nn.functional import upsample,normalize
# from ...nn.da_att import PAM_Module
# from ...nn.da_att import CAM_Module

import torchvision.models as models
from ..resnet import resnet101


# __all__ = ['DANet', 'get_danet']
__all__ = ['DANet']

class DANet(nn.Module):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, backbone, aux=False, se_loss=False, **kwargs):
        super(DANet, self).__init__()
        self.backbone = resnet101(pretrained=False)
        self.head = DANetHead(2048, nclass)

    def forward(self, x):
        imsize = x.size()[2:]
        # _, _, c3, c4 = self.base_forward(x)
        x = self.backbone.conv1(x)
        # print('x', x.size())
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        # print('c1', c1.size())
        c2 = self.backbone.layer2(c1)
        # print('c2',c2.size())
        c3 = self.backbone.layer3(c2)
        # print('c3', c3.size())
        c4= self.backbone.layer4(c3)
        # print('c4', c4.size())



        # print(x.shape)

        x = self.head(c4)
        # x = list(x)
        # x[0] = upsample(x[0], imsize, **self._up_kwargs)
        # x[1] = upsample(x[1], imsize, **self._up_kwargs)
        # x[2] = upsample(x[2], imsize, **self._up_kwargs)

        # outputs = [x[0]]
        # outputs.append(x[1])
        # outputs.append(x[2])
        outputs = F.interpolate(x, imsize, mode='bilinear', align_corners=True)
        return outputs # tuple(outputs)

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)  # [B, 512, H, W]
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)  # [B, 512, H, W]

        feat2 = self.conv5c(x)  # [B, 512, H, W]
        sc_feat = self.sc(feat2)  # [B, 512, H, W]
        sc_conv = self.conv52(sc_feat)  # [B, 512, H, W]
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        return sasc_output # tuple(output)


# def get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
#            root='~/.encoding/models', **kwargs):
#     r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
#     <https://arxiv.org/abs/1809.02983.pdf>`
#     """
#     acronyms = {
#         'pascal_voc': 'voc',
#         'pascal_aug': 'voc',
#         'pcontext': 'pcontext',
#         'ade20k': 'ade',
#         'cityscapes': 'cityscapes',
#     }
#     # infer number of classes
#     from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
#     model = DANet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
#     if pretrained:
#         #from ..model_store import model_store import get_model_file
#         from .model_store import get_model_file
#         model.load_state_dict(torch.load(
#             get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
#             strict=False)
#     return model

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)    # [m_batchsize, W*H, 64]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)   # [m_batchsize, 64, W*H]
        energy = torch.bmm(proj_query, proj_key)  # [m_batchsize, W*H, W*H]
        attention = self.softmax(energy)   # [m_batchsize, W*H, W*H]
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # [m_batchsize, 512, W*H]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)  # [B, 512, W*H]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # [B, W*H, 512]
        energy = torch.bmm(proj_query, proj_key)  # [B, 512, 512]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # # [B, 512, 512]
        attention = self.softmax(energy_new)  # [B, 512, 512]
        proj_value = x.view(m_batchsize, C, -1) # [B, 512, W*H]

        out = torch.bmm(attention, proj_value)  # [B, 512, W*H]
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
