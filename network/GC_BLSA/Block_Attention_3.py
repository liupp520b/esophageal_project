import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class KQVConv(nn.Module):
    def __init__(self, in_dim):
        super(KQVConv, self).__init__()
        self.conv_k = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.bn_k = nn.BatchNorm2d(in_dim//8)
        self.conv_q = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.bn_q = nn.BatchNorm2d(in_dim // 8)
        self.conv_v = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(in_dim//8)

    def forward(self, x):
        K = self.bn_k(self.conv_k(x))
        Q = self.bn_q(self.conv_q(x))
        V = self.bn_v(self.conv_v(x))
        return K, Q, V

class self_attention(nn.Module):
    def __init__(self, dim, v_dim):
        super(self_attention, self).__init__()

        self.dim = dim
        self.v_dim = v_dim
        self.aff_bn = nn.BatchNorm2d(256)

    def forward(self, K, Q, V):
        # x_r = x_r
        b,_,h,w = K.size()
        # print("Q:",Q.size())
        Q = Q.contiguous().view(b, self.dim, h*w).permute(0, 2, 1)  # (b, h*w, 256)
        K = K.contiguous().view(b, self.dim, h*w)  # (b, 256, h*w)
        V = V.contiguous().view(b, self.v_dim, h*w).permute(0, 2, 1)  # (b, h*w, 2048)
        att = F.softmax(torch.matmul(Q, K), dim=1)  # (b,h*w,h*w)
        out = torch.matmul(att, V).contiguous().view(b, h, w, self.v_dim)
        out = out.contiguous().permute(0, 3, 1, 2)
        # out = out+x_r
        return out


class Block_Attention_3(nn.Module):
    def __init__(self, in_dim, feature_size, patch_size):
        super(Block_Attention_3, self).__init__()
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.feature_height, self.feature_width = pair(feature_size)
        self.patch_height, self.patch_width = pair(patch_size)
        self.to_kqv = KQVConv(in_dim=in_dim)
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.feature_width//self.patch_width, self.feature_height//self.patch_height),
                                     stride=(self.feature_width//self.patch_width, self.feature_height//self.patch_height))
        self.max_pool = nn.MaxPool2d(kernel_size=(self.feature_width//self.patch_width, self.feature_height//self.patch_height),
                                     stride=(self.feature_width//self.patch_width, self.feature_height//self.patch_height))
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.self_attention = self_attention(dim=in_dim // 8, v_dim=in_dim // 8)
        # self.position_bias = nn.Parameter(torch.randn(16,16))
        self.position = nn.Parameter(torch.randn(4, 256, 16, 16))
        # self.K_position = nn.Parameter(torch.randn(4, 256, 16, 16))
        # self.V_position = nn.Parameter(torch.randn(4, 256, 16, 16))
        self.out_bn = nn.BatchNorm2d(256)

    def forward(self, x):
        b, c, h, w = x.size()
        x_in = x.clone()
        x_loc = torch.zeros(b,256,self.feature_height,self.feature_width).cuda()
        stride = self.patch_size
        # q_loc = torch.zeros(b,256,self.patch_height,self.patch_width).cuda()
        # print("q_loc:",q_loc.size())
        q_loc = torch.zeros(b,256,self.feature_height,self.feature_width).cuda()
        # print("q_loc:",q_loc.size())
        # J_loc = q_loc
        k_loc = torch.zeros(b,256,self.feature_height,self.feature_width).cuda()
        v_loc = torch.zeros(b,256,self.feature_height,self.feature_width).cuda()
        Incidence = self.down_conv(x)
        Incidence_x = F.softmax(Incidence, dim=1)
        # max_incidence = self.max_pool(Incidence)
        # avg_incidence = self.avg_pool(Incidence)
        # Incidence_matrix = max_incidence + avg_incidence
        # print('Incidence_matrix',Incidence_matrix.size())
        ''' position embeding '''
        # all_embading = torch.index_select(self.relative, 1, self.flatten_index).view(-1, 16,16)
        # q_embeding, k_embeding, v_embeding = torch.split(all_embading, [1, 1, 2], dim=0)

        for i in range(0, 4):
            for j in range(0, 4):
                x_p = x_in[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)]  # 16*16*2048
                k, q, v = self.to_kqv(x_p)
                q_loc[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)] = q   # (B, 256, 16, 16)
                k_loc[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)] = k # (B, 256, 16, 16)
                v_loc[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)] = v # (B, 2048, 16, 16)
        J_loc = torch.zeros(b,256,self.feature_height,self.feature_width).cuda()
        # print("out:",out.size())
        for m in range(0,16):
            Incidence_row = Incidence_x[:, :, :, m].view(b, c//8, 4 ,4)
            for i in range(0, 4):
                for j in range(0, 4):
                    # J_loc[:,:,4*i:4*(i+1),4*j :4*(j+1)] = q_loc[:,:,4*i:4*(i+1),4*j :4*(j+1)]*Incidence_row[:,:,i,j].view(b, c//8,1,-1)
                    J_loc[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)] = J_loc[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)] + q_loc[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)]*Incidence_row[:,:,i,j].view(b, c//8,1,-1)
        # print("1111:",J_loc.size())
        for i in range(0, 4):
            for j in range(0, 4):
                # x_r = x_in[:, :, 4 * i:4 * (i + 1), 4 * j:4 * (j + 1)]
                pos = self.position[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)]  # (b, 256, 4 ,4)
                # print("J_pos", J_pos.size())
                # K_pos = self.K_position[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)]  # (b, 256, 4 ,4)
                # # print("K_pos", K_pos.size())
                # V_pos = self.V_position[:,:,stride*i:stride*(i+1),stride*j :stride*(j+1)]  # (b, 2048, 4 ,4)
                # print("V_pos", V_pos.size())
                J = J_loc[:, :, stride * i:stride * (i + 1), stride * j:stride * (j + 1)] + pos
                # print("J", J.size())
                K = k_loc[:, :, stride * i:stride * (i + 1), stride * j:stride * (j + 1)] + pos
                # print("K", K.size())
                V = v_loc[:, :, stride * i:stride * (i + 1), stride * j:stride * (j + 1)] + pos
                output = self.self_attention(J, K, V)
                out = self.out_bn(output)
                x_loc[:, :, stride * i:stride * (i + 1), stride * j:stride * (j + 1)] = out
        x = torch.cat([x,x_loc], dim=1)
        return x

# block = Block_Attention(in_dim=2048, feature_size=64, patch_size=16)
# x = torch.randn((8, 2048, 64, 64))
# x = block(x)
# print("x:", x.size())