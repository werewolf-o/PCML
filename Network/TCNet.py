import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t
# from model.basicseg.main_blocks import CSI, DPCF
# from mmengine.model import BaseModule
from os.path import join
from model.EISNet.modules import AEIM, MRFM
from timm.models.layers import DropPath
from Conv2Former.conv2former import conv2former_n
# from Dformer.DFormer2 import DFormer_Small
from backbone.Dformer.DFormer2 import DFormer_Small
from model2.LXNet1.mona_with_select import MonaOp
import numbers
import numpy as np
from typing import Callable
from functools import partial
import math
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from FreqFusion import FreqFusion
# from model.xh.EMA import EMA
# from model2.EISNet_改_LXNet_graph2.ASPP import ASPP
# from pytorch_wavelets import DWTForward
# from model2.EISNet_改_LXNet_shiyan.FourierUnit_modified import FourierUnit_modified
# from model2.EISNet.decoder.multilevel_interaction_attention import MIA
# from Segformer.mix_transformer3 import mit_b0



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

class merge_out(nn.Module):
    def __init__(self, in1,in2):
        super(merge_out, self).__init__()
        self.last_conv1 = BasicConv2d(in2, in1, kernel_size=1, stride=1, padding=0)
        self.last_conv2 = BasicConv2d(in1*2, 64, kernel_size=1, stride=1, padding=0)
        self.last_conv3 = BasicConv2d(in1, in1, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x, y):
        y = self.last_conv1(y)
        y = self.up(y)
        x = torch.cat((x,y),dim=1)
        out = self.last_conv2(x)
        return out

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = nn.Conv2d(in_channels, out_channels, 3 ,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_last = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_last2 = nn.Conv2d(out_channels, num_class, kernel_size=1, bias=False)

    def forward(self, up_feature0,up_feature1,up_feature2,up_feature3):
        x = torch.cat([up_feature0, up_feature1, up_feature2, up_feature3], dim=1)
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
        return x, logit

class ffn(nn.Module):
    def __init__(self, num_feat, ffn_expand=2):
        super(ffn, self).__init__()

        dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(num_feat, dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, num_feat, kernel_size=1, padding=0, stride=1)

        # self.sg = SimpleGate()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        # x = x * self.sca(x)
        x = self.conv3(x)
        return x

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class UP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.act2 = nn.GELU()

    def forward(self,x):
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

class AP_MP(nn.Module):
    def __init__(self, stride=2):
        super(AP_MP, self).__init__()
        self.sz = stride
        self.gapLayer = nn.AvgPool2d(3, stride=1,padding=1)
        self.gmpLayer = nn.MaxPool2d(3, stride=1,padding=1)

    def forward(self, x):
        apimg = self.gapLayer(x)
        mpimg = self.gmpLayer(x)
        byimg = torch.norm(abs(apimg - mpimg), p=2, dim=1, keepdim=True)
        return byimg

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

class Adapter(nn.Module):
    def __init__(self, dim) -> None:
        super(Adapter, self).__init__()
        self.monaOp = MonaOp(dim)

    def forward(self, x):
        x =  self.monaOp(x)
        return x

class uncertainty_generation(nn.Module):
    def __init__(self, dim):
        super(uncertainty_generation, self).__init__()
        # self.imgsize = imgsize
        # self.soft_split = nn.Unfold(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))                     #image-->tokens
        # self.soft_fuse = nn.Fold(output_size=self.imgsize, kernel_size=(4, 4),
        #                           stride=(4, 4), padding=(0, 0))                                           #tokens-->image
        self.mean_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0),
        )

        self.std_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0),
        )

        kernel = torch.ones((7, 7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def reparameterize(self, mu, logvar, k):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()  # type:
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z

    def forward(self, x):
        # x = self.soft_fuse(x.transpose(-2, -1))

        mean = self.mean_conv(x)
        std = self.std_conv(x)

        prob = self.reparameterize(mean, std, 3)

        prob_out = self.reparameterize(mean, std, 50)
        prob_out = torch.sigmoid(prob_out)
        uncertainty = prob_out.var(dim=1, keepdim=True).detach()
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        uncertainty = (1 - uncertainty) * x
        # uncertainty = self.soft_split(uncertainty).transpose(-2, -1)
        return prob, uncertainty

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.ccoo = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TDAtt(nn.Module):
    def __init__(self, dim):
        super(TDAtt, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (15, 1), padding=(7, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u



class TCNet(nn.Module):
    def __init__(self, num_classes=3, embed_dim=64,
                 depths=[2, 2, 2, 2],
                 mlp_ratio=4, drop_path_rate=0.1, patch_norm=True, **kwargs):
        super().__init__()
        self.num_class = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()

        self.ffm = FeatureFusionModule(embed_dim * 4, embed_dim, num_class=self.num_class)

        self.con = nn.Conv2d(320,256,1)

        self.label_r = DFormer_Small()
        self.label_r._init_weights("/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/backbone/Dformer/DFormer_Small.pth.tar")

        self.conv1 = nn.Conv2d(3,1,1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.soft = nn.Softmax(dim=1)

        self.MIA1 = merge_out(385,641)   # 18, 32
        self.MIA2 = merge_out(257,64)   # 36, 64
        self.MIA3 = merge_out(193,64)   # 72, 128
        self.MIA4 = CONVM(64, 64)   # 72, 128

        self.aux1 = Fuse(64, 32, 4, num_class=3)
        self.aux2 = Fuse(64, 32, 4, num_class=3)
        self.aux3 = Fuse(64, 32, 4, num_class=3)
        self.aux4 = Fuse(64, 32, 4, num_class=3)

        self.Adapter1 = Adapter(3)
        self.Adapter2 = Adapter(64)
        self.Adapter3 = Adapter(128)
        self.Adapter4 = Adapter(256)

        self.upsample16 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.uncertainty = uncertainty_generation(512)
        self.AP_MP = AP_MP()

        self.TDAtt0 = TDAtt(128)
        self.TDAtt1 = TDAtt(128)
        self.TDAtt2 = TDAtt(128)
        self.TDAtt3 = TDAtt(128)

        self.conv_cat0 = nn.Sequential(
            nn.Conv2d(576, 128, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )
        self.conv_cat1 = nn.Sequential(
            nn.Conv2d(640, 128, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )
        self.conv_cat2 = nn.Sequential(
            nn.Conv2d(768, 128, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )
        self.conv_cat3 = nn.Sequential(
            nn.Conv2d(1024, 128, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.ax1 = nn.Sequential(
            nn.Conv2d(193, 64, 1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )
        self.ax2 = nn.Sequential(
            nn.Conv2d(257, 128, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )
        self.ax3 = nn.Sequential(
            nn.Conv2d(385, 256, 1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )
        self.ax4 = nn.Sequential(
            nn.Conv2d(641, 512, 1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )


    def encode_decode(self, x, x_e):
        # stage_r = self.label_r.forward(x, y)
        # stage_d = self.label_d.forward(y, x)
        # stage_r = self.label_r.forward_features(x)
        # stage_d = self.label_d.forward_features(y)
        outs = []
        y = x

        # Dformer

        if x_e is None:
            x_e = x
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(x_e.shape) == 3:
            x_e = x_e.unsqueeze(0)

        xx_e = x_e
        x_e = x_e[:, 0, :, :].unsqueeze(1)
        xx = x[:, 0, :, :].unsqueeze(1)

        # Stage 1

        x = self.Adapter1(x)
        r0,d0 = self.label_r.blk0(x, x_e)
        outs.append(r0)

        # Stage 2
        r0 = self.Adapter2(r0)
        r1,d1 = self.label_r.blk1(r0,d0)
        outs.append(r1)

        # Stage 3
        r1 = self.Adapter3(r1)
        r2,d2 = self.label_r.blk2(r1,d1)
        outs.append(r2)

        # Stage 4
        r2 = self.Adapter4(r2)
        r3 = self.label_r.blk3(r2,d2)
        outs.append(r3)

        prob, uncertainty = self.uncertainty(r3)

        r0 = self.conv_cat0(torch.cat([self.upsample8(uncertainty),r0],dim=1))
        r1 = self.conv_cat1(torch.cat([self.upsample4(uncertainty),r1],dim=1))
        r2 = self.conv_cat2(torch.cat([self.upsample2(uncertainty),r2],dim=1))
        r3 = self.conv_cat3(torch.cat([uncertainty,r3],dim=1))

        r0 = self.TDAtt0(r0)
        r1 = self.TDAtt1(r1)
        r2 = self.TDAtt2(r2)
        r3 = self.TDAtt3(r3)

        r0 = torch.cat([self.AP_MP(r0),r0, outs[0]],dim=1)
        r1 = torch.cat([self.AP_MP(r1),r1,outs[1]], dim=1)
        r2 = torch.cat([self.AP_MP(r2),r2,outs[2]], dim=1)
        r3 = torch.cat([self.AP_MP(r3),r3,outs[3]], dim=1)



        P1 = self.ax1(r0)
        P2 = self.ax2(r1)
        P3 = self.ax3(r2)
        P4 = self.ax4(r3)


        up_feature0 = self.MIA1(r2, r3)
        up_feature1 = self.MIA2(r1, up_feature0)
        up_feature2 = self.MIA3(r0, up_feature1)
        up_feature3 = self.MIA4(up_feature2)

        up_feature0 = F.interpolate(up_feature0, size=(72, 128), mode='bilinear', align_corners=False)
        up_feature1 = F.interpolate(up_feature1, size=(72, 128), mode='bilinear', align_corners=False)


        aux1 = self.aux1(up_feature0)
        aux2 = self.aux2(up_feature1)
        aux3 = self.aux3(up_feature2)
        aux4 = self.aux4(up_feature3)

        x, logit = self.ffm(up_feature0, up_feature1, up_feature2, up_feature3)

        return x,aux1,aux2,aux3,aux4,P1,P2,P3,P4

    def forward(self, input):

        rgb = input[:, :3]
        modal_x = input[:, 3:]
        modal_x = torch.cat((modal_x, modal_x, modal_x), dim=1)

        x,aux1,aux2,aux3,aux4,P1,P2,P3,P4 = self.encode_decode(rgb,modal_x)

        return x,aux1,aux2,aux3,aux4,P1,P2,P3,P4

if __name__ == '__main__':
    model = TCNet().cuda(0)
    a = torch.randn(2, 3, 288, 512).cuda(0)
    b = torch.randn(2, 1, 288, 512).cuda(0)
    images = torch.cat((a, b), dim=1)
    out = model(images)

    for i in range(len(out)):
        print(out[i].shape)
