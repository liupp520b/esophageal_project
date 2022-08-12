import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def transform(t, c, patch_size):
    patch_list = []
    for i in t:
        b, h, w =i.size()
        # print("b:", b, "h:", h, "w:",w)
        i = i.contiguous().view(b, c, patch_size, patch_size)
        patch_list.append(i)
    return patch_list

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

class self_attention(nn.Module):
    def __init__(self, num_patch):
        super(self_attention, self).__init__()

        self.num_patch = num_patch

    def forward(self, K_set, J_set, V_set):
        att_set = []
        for i in range(self.num_patch):
            K = K_set[i]
            K_batchsize, K_chanel, K_height, K_width = K.size()
            J = J_set[i]
            J_batchsize, J_chanel, J_height, J_width = J.size()
            V = V_set[i]
            V_batchsize, V_chanel, V_height, V_width = V.size()
            K_pie = K.contiguous().view(K_batchsize, K_chanel, K_height*K_width)
            # print("K_pie:",K_pie.size())
            J_pie = J.contiguous().view(J_batchsize, J_chanel, J_height*J_width).permute(0, 2, 1)
            # print("J_pie:", J_pie.size())
            V_pie = V.contiguous().view(V_batchsize, -1, V_height*V_width)
            # print("V_pie:", V_pie.size())
            attention = F.softmax(torch.bmm(J_pie, K_pie), dim=1)
            # print("attention:", attention.size())
            out = torch.bmm(V_pie, attention).view(V_batchsize, V_chanel, V_height, V_width)
            att_set.append(out)

        return att_set


class Block_Attention(nn.Module):
    def __init__(self, in_dim, feature_size, patch_size):
        super(Block_Attention, self).__init__()

        self.feature_height, self.feature_width = pair(feature_size)
        self.patch_height, self.patch_width = pair(patch_size)

        self.patch_size = patch_size
        assert self.feature_height % self.patch_height == 0 and self.feature_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (self.feature_height // self.patch_height) * (self.feature_width // self.patch_width)
        # self.register_buffer('flatten_index', relative_index.view(-1))

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width)
        self.pos_embeding = nn.Parameter(torch.randn(1, self.num_patch, in_dim))
        self.patch_splid_index = []
        self.J_splid_index = []
        for i in range(self.num_patch):
            self.patch_splid_index.append(1)

        for i in range(self.num_patch):
            self.J_splid_index.append(self.patch_height)

        self.to_kqv = KQVConv(in_dim=in_dim)
        # self.Incidence_matrix = nn.Parameter(torch.randn(self.patch_height*self.num_patch, self.patch_height*self.num_patch))
        self.self_attention = self_attention(self.num_patch)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.max_pool = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.relu = nn.ReLU()
        self.Incidence_matrix = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # self.Incidence_matrix = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=5, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = x
        Incidence_matrix = self.Incidence_matrix(x)
        Incidence_matrix_1 = self.relu(self.avg_pool(Incidence_matrix))
        Incidence_matrix_2 = self.relu(self.max_pool(Incidence_matrix))
        Incidence_matrix = Incidence_matrix_1 + Incidence_matrix_2
        Incidence_b, Incidence_c, Incidence_h, Incidence_w = Incidence_matrix.size()
        Incidence_matrix = Incidence_matrix.view(Incidence_b, Incidence_c, Incidence_h*Incidence_w)
        # Incidence_matrix = F.interpolate(Incidence_matrix, (256, 256), mode='bilinear', align_corners=True)
        B, C, H, W = x.size()
        x = self.to_patch(x)   # (bachsize, num_patch, patch_token)  # [8, 16, 524288]
        # x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width)
        patch_set = torch.split(x, self.patch_splid_index, dim=1)
        patch_list = transform(patch_set, C, self.patch_size)
        K_set = []
        Q_set = []
        V_set = []
        for patch in patch_list:
            K, Q, V = self.to_kqv(patch)
            K_set.append(K)   # (B, C//8, patch_size, patch_size)
            Q_set.append(Q)   # (B, C//8, patch_size, patch_size)
            V_set.append(V)   # (B, C, patch_size, patch_size)
        # Incidence_block_b, Incidence_block_c = Incidence_matrix[:, :, 0].size()
        Q_block = Q_set[0]*Incidence_matrix[:,:,0].view(Incidence_b, Incidence_c, 1, -1)

        J_set = []
        for i in range(1, self.num_patch):
            # Q_block = torch.cat([Q_block, Q_set[i]*Incidence_matrix[:,:,i]], dim=2)
            Q_block = Q_block + Q_set[i]*Incidence_matrix[:,:,i].view(Incidence_b, Incidence_c, 1, -1)
        for i in range(0, self.num_patch):
            J_set.append(Q_block)
        # print("Q_block:",Q_block.size())  #torch.Size([8, 256, 256, 16])
        # J_block = torch.matmul(Incidence_matrix, Q_block)
        # print("J_block:", J_block.size())  #torch.Size([8, 256, 256, 16])
        # b_j, c_j, h_j, w_j = J_block.size()
        # J_ = torch.split(Q_block, self.J_splid_index, dim=2)

        # for i in ran:
        #     # print("i:",i.size())  #torch.Size([8, 256, 16, 16])
        #     J_set.append(i)
        attention_out = self.self_attention(K_set, J_set, V_set)
        out = attention_out[0]
        for i in range(1, self.num_patch):
            out = torch.cat([out, attention_out[i]], dim=2)
        out_batchsize, out_channek, out_height, out_width = out.size()
        out = out.contiguous().view(out_batchsize, out_channek, -1, out_width*(self.feature_width//self.patch_width))
        out = out + x1
        # out = self.outconv(out)
        return out

# block = Block_Attention(in_dim=2048, feature_size=64, patch_size=16)
# x = torch.randn((8, 2048, 64, 64))
# x = block(x)
# print("x:", x.size())