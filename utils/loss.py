import torch.nn as nn
import torch.nn.functional as F
import torch

#import tensorflow as tf
#from torch import *

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        # print(focal_loss)
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# class SoftmaxLoss(nn.Module):
#     def __init__(self, input, targets):
#         super(SoftmaxLoss, self).__init__()
#
#     def forward(self, input, targets):
#
#         exp_pred = torch.exp(input)
#         try:
#             sum_exp = torch.sum(exp_pred, dim=3, keepdim=True)
#         except:
#             sum_exp = torch.sum(exp_pred, dim=3, keepdim=True)
#         tensor_sum_exp = torch.sum(sum_exp, torch.stack([1, 1, 1, input.shape[3]]))
#         softmax_output = torch.div(exp_pred, tensor_sum_exp)
#         ce = -torch.sum(targets * torch.log(torch.clamp(softmax_output, 1e-12, 1.0)))
#
#         return ce

def SoftmaxLoss(input, targets):
    exp_pred = torch.exp(input)
    try:
        sum_exp = torch.sum(exp_pred, dim=3, keepdim=True)
    except:
        sum_exp = torch.sum(exp_pred, dim=3, keepdim=True)
    #tensor_sum_exp = torch.repeat_interleave(sum_exp, torch.stack([1, 1, 1, input.shape[3]]))
    #tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(input)[3]]))
    tensor_sum_exp = sum_exp.repeat([1, 1, 1, input.shape[3]])
    softmax_output = torch.div(exp_pred, tensor_sum_exp)
    #targets = torch.LongTensor(targets)
    ce = -torch.sum(targets * torch.log(torch.clamp(softmax_output, 1e-12, 1.0)))

    return ce

class SmoothNet_Loss(nn.Module):
    def __init__(self):
        super(SmoothNet_Loss, self).__init__()

    def forward(self, ls_fuse, ls4, ls3, ls2, ls1, targets):

        cefuse = SoftmaxLoss(ls_fuse, targets)

        ce4 = SoftmaxLoss(ls4, targets)
        ce3 = SoftmaxLoss(ls3, targets)
        ce2 = SoftmaxLoss(ls2, targets)
        ce1 = SoftmaxLoss(ls1, targets)

        # print(cefuse)

        total_ce = ce1 + ce2 + ce3 + ce4 + cefuse

        return total_ce
class CE_Loss(nn.Module):
    def __init__(self, num_class):
        super(CE_Loss, self).__init__()
        self.num_class = num_class
    def forward(self, inputs, target):
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        # temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        # temp_target = target.view(-1)
        CE_loss = nn.NLLLoss(ignore_index=self.num_class)(F.log_softmax(inputs, dim=-1), target)

        return CE_loss
class Dice_Loss(nn.Module):
    def __init__(self, beta, smooth):
        self.beta = beta
        self.smooth = smooth
    def forward(self, inputs, target):
        n, c, h, w = inputs.size()
        nt, ht, wt, ct = target.size()
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
        temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
        tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
        fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
        fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        dice_loss = 1 - torch.mean(score)
        return dice_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


