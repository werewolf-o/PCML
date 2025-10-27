import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from Segformer.mix_transformer3 import mit_b2, mit_b0,mit_b1

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

from model2.LXNet2.LEGM import LEGM,LayNormal,Att
from model2.LXNet2_xiaorong.FSAS import FSAS
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1

class UP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x

class CONVM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x

class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        # self.conv1x1_2 = nn.Conv2d(int(num_reduced_channels*4), 1, 1, 1)

        self.dilated_conv3x3 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=2, dilation=2)
        self.dilated_conv7x7 = nn.Conv2d(num_reduced_channels, num_reduced_channels, 3, 1, padding=3, dilation=3)
        self.bn = nn.BatchNorm2d(in_channels)
        self.bn_act =nn.Sequential(nn.BatchNorm2d(in_channels), nn.SiLU())

    def forward(self, feature_maps):
        feature_maps = self.bn(feature_maps)
        att = self.conv1x1_1(feature_maps)

        d1 = self.dilated_conv3x3(att)
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)

        att = torch.cat((att, d1, d2, d3), dim=1)
        att = self.bn_act(att)

        # att = self.conv1x1_2(att)

        return att

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_last = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_last2 = nn.Conv2d(out_channels, num_class, kernel_size=1, bias=False)

    def forward(self, up_feature):
        x = torch.cat(up_feature, dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        feature = self.act1(self.bn1(feature))

        x = self.avgpool(x)
        x = self.act2(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        logit = self.conv_last(x)
        x = F.interpolate(logit, scale_factor=2, mode='bilinear')
        x = self.conv_last2(x)
        return x,logit

class Fuse(nn.Module):

    def __init__(self, in_, out, scale, num_class):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(out, num_class, 1, bias=False)

    def forward(self, up_inp):
        outputs = F.interpolate(up_inp, scale_factor=self.scale, mode='bilinear')
        outputs = self.conv1(outputs)
        outputs = self.activation(outputs)
        outputs = self.conv2(outputs)
        return outputs

class downs(nn.Module):
    def __init__(self, intc, outc):
        super().__init__()
        self.conv1 = nn.Conv2d(intc, outc, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias =False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias = bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

class LECSFormer(nn.Module):

    def __init__(self,
                 img_size=224, patch_size=4, in_channels=3, num_classes=3, embed_dim=64,
                 depths=[2, 2, 2, 2],
                 norm_layer=nn.LayerNorm, patch_norm=True,
                  **kwargs):
        super().__init__()

        self.num_class = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # Decoder
        self.decode1_1 = UP(embed_dim * 8, embed_dim * 4)
        self.decode1_2 = UP(embed_dim * 4, embed_dim * 2)
        self.decode1_3 = UP(embed_dim * 2, embed_dim)

        self.decode2_1 = UP(embed_dim * 4, embed_dim * 2)
        self.decode2_2 = UP(embed_dim * 2, embed_dim)

        self.decode3_1 = UP(embed_dim * 2, embed_dim)

        self.decode4_1 = CONVM(embed_dim, embed_dim)

        self.ffm = FeatureFusionModule(embed_dim * 4, embed_dim, num_class=self.num_class)

        self.aux1 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)
        self.aux2 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)
        self.aux3 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)
        self.aux4 = Fuse(embed_dim, embed_dim, 4, num_class=self.num_class)

        self.apply(self._init_weights)
        self.label_r = mit_b1()
        self.label_d = mit_b1()
        self.label_r.init_weights("/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/Segformer/mit_b1.pth")
        self.label_d.init_weights("/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/Segformer/mit_b1.pth")
        self.con = nn.Conv2d(320,256,1)

        self.conv1 = nn.Conv2d(3,1,1)


        self.legm1 = LEGM(network_depth=4, dim=64, num_heads=8, mlp_ratio=4.0, norm_layer=LayNormal, mlp_norm=False,
                           window_size=8,
                           shift_size=4, use_attn=True, conv_type='DWConv')
        self.legm2 = LEGM(network_depth=4, dim=128, num_heads=8, mlp_ratio=4.0, norm_layer=LayNormal, mlp_norm=False,
                           window_size=8, shift_size=4, use_attn=True, conv_type='DWConv')
        self.legm3 = LEGM(network_depth=4, dim=256, num_heads=8, mlp_ratio=4.0, norm_layer=LayNormal, mlp_norm=False,
                           window_size=8, shift_size=4, use_attn=True, conv_type='DWConv')
        self.legm4 = LEGM(network_depth=4, dim=512, num_heads=8, mlp_ratio=4.0, norm_layer=LayNormal, mlp_norm=False,
                           window_size=8, shift_size=4, use_attn=True, conv_type='DWConv')

        self.last_conv1 = BasicConv2d(256 + 512, 256, kernel_size=1, stride=1, padding=0)
        self.last_conv2 = BasicConv2d(256 + 128, 128, kernel_size=1, stride=1, padding=0)
        self.last_conv3 = BasicConv2d(128 + 64, 64, kernel_size=1, stride=1, padding=0)

        self.last_c3 = BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.last_c2 = BasicConv2d(64+128, 64, kernel_size=3, stride=2, padding=1)
        self.last_c1 = BasicConv2d(256+64, 64, kernel_size=3, stride=2, padding=1)
        self.last_c0 = BasicConv2d(512+64 , 512, kernel_size=1, stride=1, padding=0)

        self.Tfusion1 = CSPRepLayer(64, 64, 3)
        self.Tfusion2 = CSPRepLayer(64, 64, 3)
        self.Tfusion3 = CSPRepLayer(512, 512, 3)

        self.Pfusion1 = CSPRepLayer(256, 256, 3)
        self.Pfusion2 = CSPRepLayer(128, 128, 3)
        self.Pfusion3 = CSPRepLayer(64, 64, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(960, 254, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(254, 4, 1, padding=0, bias=True),
        )
        self.conv_edg = nn.Conv2d(960, 128, kernel_size=1, stride=1, padding=0)
        self.fsas = FSAS(128)
        self.to64 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.to128 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.to256 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.to512 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.atten1 = LocalSpatialAttention(512,128)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_up_features(self, stage):
        x_1_1 = self.decode1_1(stage[3])
        x_1_2 = self.decode1_2(x_1_1)
        x_1_3 = self.decode1_3(x_1_2)
        # stage[2] =self.con(stage[2])#################
        x_2_1 = self.decode2_1(torch.add(stage[2], x_1_1))
        x_2_2 = self.decode2_2(torch.add(x_2_1, x_1_2))
        x_3_1 = self.decode3_1(stage[1] + x_2_1 + x_1_2)

        x_4_1 = self.decode4_1(stage[0] + x_1_3 + x_2_2 + x_3_1)
        return [x_1_3, x_2_2, x_3_1, x_4_1]

    def encode_decode(self, x, y):
        stage_r = self.label_r.forward_features(x)
        stage_d = self.label_d.forward_features(y)
        stage = []

        stage_r[2] = self.con(stage_r[2])

        stage_r[3] = stage_r[3]
        stage_r[2] = stage_r[2]
        stage_r[1] = stage_r[1]
        stage_r[0] = stage_r[0]

        stage_d3 = stage_d[3]
        stage_d2 = self.con(stage_d[2])
        stage_d1 = stage_d[1]
        stage_d0 = stage_d[0]

        E0 = self.last_c3(stage_d0)
        E1 = self.Tfusion1(self.last_c2(torch.cat([E0,stage_r[1]],dim=1)))
        E2 = self.Tfusion2(self.last_c1(torch.cat([E1, stage_d2], dim=1)))
        E3 = self.Tfusion3(self.last_c0(torch.cat([E2, stage_r[3]], dim=1)))

        E3 = E3 + stage_d3
        K1 = F.interpolate(E3, scale_factor=2, mode='bilinear', align_corners=False)
        KK2 = self.Pfusion1(self.last_conv1(torch.cat([K1,stage_r[2]],dim=1)))

        K2 = F.interpolate(KK2, scale_factor=2, mode='bilinear', align_corners=False)
        KK3 = self.Pfusion2(self.last_conv2(torch.cat([K2, stage_d1], dim=1)))

        K3 = F.interpolate(KK3, scale_factor=2, mode='bilinear', align_corners=False)
        KK4 = self.Pfusion3(self.last_conv3(torch.cat([K3, stage_r[0]], dim=1)))

        E3 = self.legm4(E3)
        KK2 = self.legm3(KK2)
        KK3 = self.legm2(KK3)
        KK4 = self.legm1(KK4)

        x_avg1 = self.avg_pool(E3)
        x_avg2 = self.avg_pool(KK2)
        x_avg3 = self.avg_pool(KK3)
        x_avg4 = self.avg_pool(KK4)
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3, x_avg4], dim=1)
        attention_score = self.ca(fea_avg)
        w1, w2, w3,w4 = torch.chunk(attention_score, 4, dim=1)

        E3 = E3 * w1
        KK2 = KK2 * w2
        KK3 = KK3 * w3
        KK4 = KK4 * w4

        atten = self.atten1(stage_r[3])

        T1 = atten + E3
        T2 = self.to256(F.interpolate(atten, KK2.shape[2:], mode='bilinear'))+ KK2
        T3 = self.to128(F.interpolate(atten, KK3.shape[2:], mode='bilinear'))+ KK3
        T4 = self.to64(F.interpolate(atten, KK4.shape[2:], mode='bilinear'))+ KK4

        P1 = T1
        P2 = T2
        P3 = T3
        P4 = T4

        stage.append(P4)
        stage.append(P3)
        stage.append(P2)
        stage.append(P1)

        up_feature = self.forward_up_features(stage)
        x,logit = self.ffm(up_feature)
        aux1 = self.aux1(up_feature[0])
        aux2 = self.aux2(up_feature[1])
        aux3 = self.aux3(up_feature[2])
        aux4 = self.aux4(up_feature[3])

        return x, aux1,aux2,aux3,aux4,P4,P3,P2,P1

    def forward(self, input):

        rgb = input[:, :3]
        modal_x = input[:, 3:]
        modal_x = torch.cat((modal_x, modal_x, modal_x), dim=1)

        x, aux1,aux2,aux3,aux4,P4,P3,P2,P1 = self.encode_decode(rgb,modal_x)

        return x


if __name__ == '__main__':
    model = LECSFormer()
    a = torch.rand(2, 3, 288, 512)
    b = torch.randn(2, 1, 288, 512)
    images = torch.cat([a,b], dim=1)
    out = model(images)

    for i in range(len(out)):
        print(out[i].shape)


    # from util.util import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = LECSFormer()
    #     flops, params = get_model_complexity_info(net, (4, 288, 512), as_strings=True, print_per_layer_stat=False)
    #     print('Flops: ' + flops)
    #     print('Params: ' + params)
    # #
    # compute_speed(net, input_size=(1, 4, 288, 512), iteration=500)




















