import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from thop import profile
from torch.autograd import Variable
affine_par = True
import functools
from .block import block
import torchvision.models as models
import sys, os



# from utils.pyt_utils import load_model

from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4     # 512

        self.conva_0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        # self.conva_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                              InPlaceABNSync(inter_channels))
        # self.cca_0 = CrissCrossAttention(inter_channels)
        # self.cca_1 = CrissCrossAttention_1(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        ######################### 原始 CCnet ##########################
        output = self.conva_0(x)    # 97x97x512
        for i in range(recurrence):
            output = self.cca_0(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], dim=1))
        ##############################################################
        # output_0 = self.conva_0(x)
        # output_1 = self.conva_1(x)
        # for i in range(recurrence):
        #     output0, output1 = self.cca_1(output_0, output_1)
        #     output_0 = output0 + output_0
        #     output_1 = output1 + output_1
        # output = output_1 + output_0
        # output = self.convb(output)
        # output = self.bottleneck(torch.cat([x, output], dim=1))
        ################################################################
        # output_0 = output
        # output_1 = output
        # for i in range(recurrence):
        #     output_0, output_1 = self.cca(output_0, output_1)
        # output = output_0 + output_1
        # output = self.convb(output)
        #
        # output = self.bottleneck(torch.cat([x, output], 1))
        #################################################################
        return output

class Non_local(nn.Module):
    def __init__(self, num_classes):
        super(Non_local, self).__init__()
        layers = [3, 4, 23, 3]
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # change
        self.backbone = models.resnet101(pretrained=True)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])    # (64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)  # (128, 4)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2, dilation=4, multi_grid=(1,1,1))
        self.non_local = block(2048)


        self.dsn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.outConv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0),
        )
        # self.criterion = criterion


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * Bottleneck.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(Bottleneck(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # _, _, h, w = x.size()
        # print("x: ", x.size())
        x = self.relu1(self.bn1(self.conv1(x)))   # 385x385x64
        # print("conv1: ",x.size())
        x = self.relu2(self.bn2(self.conv2(x)))   # 385x385x64
        # print("conv2: ", x.size())
        x = self.relu3(self.bn3(self.conv3(x)))   # 385x385x128
        # print("conv3: ", x.size())
        x = self.maxpool(x)    # 193x193x128
        # print("maxpool: ", x.size())
        x = self.layer1(x)     # 193x193x128
        # print("layer1: ", x.size())
        x = self.layer2(x)     # 97x97x512
        # print("layer2: ", x.size())
        x = self.layer3(x)     # 97x97x1024
        # print("layer3: ", x.size())
        # x_dsn = self.dsn(x)    # 97x97x
        # print("x_dsn:",x_dsn.size())
        c4 = self.layer4(x)     # 97x97x2048
        # print("layer4: ", x.size())
        #################### ccnet#################
        x = self.non_local(c4)    #  (97x97x5, 0)
        x_dsn = self.dsn(c4)
        x_dsn = F.interpolate(x_dsn, (512, 512), mode='bilinear', align_corners=True)
        # print("x",x.size())
        x = self.outConv(x)
        x = F.interpolate(x, (512, 512), mode='bilinear', align_corners=True)
        out = [x, x_dsn]

        return out

        # if self.criterion is not None and labels is not None:
        #     return self.criterion(outs, labels)
        # else:
        #     return outs


# def Seg_Model(num_classes, criterion=None, pretrained_model=None, recurrence=0, **kwargs):
#     model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, criterion, recurrence)
#
#     if pretrained_model is not None:
#         model = load_model(model, pretrained_model)
#
#     return model

# if __name__ == '__main__':
#     print(Seg_Model(5))