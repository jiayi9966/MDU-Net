import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import LayerNorm
from .SE_weight_module import SEWeightModule
from functools import partial
#from .basicnet import MutualNet
import math
#import Constants
from .vit_seg_modeling_resnet_skip import ResNetV2
from .deform_conv_v2 import DeformConv2d

nonlinearity = partial(F.relu, inplace=True)

#添加多尺度注意力
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class double_deform_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out


class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.se = SEWeightModule(self.channels_single)
        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        #p1 = self.p1(p1_input)
        #p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_input
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((self.se(p1_input)*p1_input, self.se(p2_dc)*p2_dc, self.se(p3_dc)*p3_dc, self.se(p4_dc)*p4_dc), 1))

        return ce

class GCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM, self).__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),#64channel
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            #NonLocalBlock(out_channels)
            SA_Block(64)
        ))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], kernel_size=1),#channel
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
            #print("global_context1", global_context[i].shape)
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)
        #print("global_context",global_context.shape)
        output = []
        for i in range(len(self.GCoutmodel)):
            output.append(self.GCoutmodel[i](global_context))

        return output
class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class GCM_msf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM_msf, self).__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),#64channel
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            #NonLocalBlock(out_channels)
            SA_Block(64)
        ))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], kernel_size=1),#channel
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), )
        # nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()
        self.SE1 = oneConv(64, 64, 1, 0, 1)
        self.SE2 = oneConv(64, 64, 1, 0, 1)
        self.SE3 = oneConv(64, 64, 1, 0, 1)
        self.SE4 = oneConv(64, 64, 1, 0, 1)
    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
            #print("global_context1", global_context[i].shape)
        y3_weight = self.SE4(self.gap(global_context[0]))
        y2_weight = self.SE3(self.gap(global_context[1]))
        y1_weight = self.SE2(self.gap(global_context[2]))
        y0_weight = self.SE1(self.gap(self.GCmodule[-1](x)))
        weight = torch.cat([y0_weight, y1_weight, y2_weight, y3_weight], 2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        y3_weight = torch.unsqueeze(weight[:, :, 3], 2)
        # x_att = y0_weight * global_context[0] + y1_weight * global_context[1] + y2_weight * global_context[2] + y3_weight * global_context[3]
        global_context = torch.cat((y0_weight * global_context[0],
                            y1_weight * global_context[1],
                            y2_weight * global_context[2],
                            y3_weight * self.GCmodule[-1](x)), dim=1)
        #print("global_context",global_context.shape)
        output = []
        for i in range(len(self.GCoutmodel)):
            output.append(self.GCoutmodel[i](global_context))
        return output
#+MLFC+CNN_GCM_msf
class UCRNet10_6(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2,n_filts = 64):
        super(UCRNet10_6, self).__init__()

        filters = [64, 128, 256, 512]
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.vit=ResNetV2((3, 4, 9),1)
        self.gcm = GCM_msf(512,64)
        self.mlfc1 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc2 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc3 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)

        #self.dblock = Dblock(512)
        self.decoder4 = Decoder4(in_ch1=512, out_ch=256, in_ch2=256,in_ch3=256)#in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder4(in_ch1=256, out_ch=128, in_ch2=128,in_ch3=512)
        self.decoder2 = Decoder4(in_ch1=128, out_ch=64, in_ch2=64,in_ch3=256)
        self.decoder1 = Decoder4(in_ch1=64, out_ch=64, in_ch2=64,in_ch3=64)
        #self.deconv4 = nn.ConvTranspose2d(512, filters[3], 3, stride=2, padding=1, output_padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)#zhuanzhi
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        c, f = self.vit(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        #x = self.firstact(x)
        x = self.firstmaxpool(x_0)
        e1 = self.encoder1(x)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)



        g = self.gcm(e4)
        x_0, e1, e2, e3 = self.mlfc1(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc2(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc3(x_0, e1, e2, e3)
        # Decoder
        up4 = self.decoder4(e4,e3,g[0],e3)
        up3 = self.decoder3(up4,e2,g[1],f[0])
        up2 = self.decoder2(up3,e1,g[2],f[1])
        up1 = self.decoder1(up2,x_0,g[3],f[2]) # f[2]:(2, 1024, 38, 38)

        out = self.finaldeconv1(up1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 3, dim * 3, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip, x2):
        output = torch.cat([x, skip, x2], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip) + self.conv3(x2)
        att = self.nonlin(att)
        output = output * att
        return output
def replace_conv_with_snakeconv(self, block):
    for name, module in block.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(block, name, DeformConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias
            ))
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            self.replace_conv_with_snakeconv(module)
    return block


#+mlfc+cnn+DFF
class UCRNet10_5(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2,n_filts = 64):
        super(UCRNet10_5, self).__init__()

        filters = [64, 128, 256, 512]
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.vit = ResNetV2((3, 4, 9), 1)
        self.gcm = GCM(512, 64)
        self.mlfc1 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc2 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc3 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)

        # self.dblock = Dblock(512)
        self.decoder4 = Decoder3(in_ch1=512, out_ch=256, in_ch2=256, in_ch3=256)  # in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder3(in_ch1=256, out_ch=128, in_ch2=128, in_ch3=512)
        self.decoder2 = Decoder3(in_ch1=128, out_ch=64, in_ch2=64, in_ch3=256)
        self.decoder1 = Decoder3(in_ch1=64, out_ch=64, in_ch2=64, in_ch3=64)
        # self.deconv4 = nn.ConvTranspose2d(512, filters[3], 3, stride=2, padding=1, output_padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # zhuanzhi
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        c, f = self.vit(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        # x = self.firstact(x)
        x = self.firstmaxpool(x_0)
        e1 = self.encoder1(x)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)

        g = self.gcm(e4)
        x_0, e1, e2, e3 = self.mlfc1(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc2(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc3(x_0, e1, e2, e3)
        # Decoder
        up4 = self.decoder4(e4, e3, g[0], e3)
        up3 = self.decoder3(up4, e2, g[1], f[0])
        up2 = self.decoder2(up3, e1, g[2], f[1])
        up1 = self.decoder1(up2, x_0, g[3], f[2])  # f[2]:(2, 1024, 38, 38)

        out = self.finaldeconv1(up1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out



class Decoder5(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch1, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nonlinearity
        #self.coatt = CCA(F_g=out_ch // 2, F_x= in_ch2 // 2)
        # self.ce = Context_Exploration_Block(in_ch2)
        self.conv = DoubleConv(out_ch + in_ch2 + in_ch2, out_ch)
        #self.residual = Residual(out_ch + in_ch2 + in_ch2, out_ch)
        self.se = SEWeightModule(out_ch + in_ch2 + in_ch2)
        self.dff = DFF(in_ch2)
    def forward(self, x1, x2, x3):
        x1 = self.deconv(x1)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        #skip_x_att = self.coatt(g=x1, x=x2)
        # input is CHW
        # channel attetion
        # c2 = self.ce(x2)
        # c = torch.cat([x1, x2, x3], dim=1)
        output=self.dff(x1, x2, x3)
        # w = self.se(output)*output
        fuse = self.conv(output)
        return fuse



#+MLFC+CNN
class UCRNet10_3(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2,n_filts = 64):
        super(UCRNet10_3, self).__init__()

        filters = [64, 128, 256, 512]
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.vit=ResNetV2((3, 4, 9),1)
        self.gcm = GCM(512,64)
        self.mlfc1 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc2 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc3 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)

        #self.dblock = Dblock(512)
        self.decoder4 = Decoder4(in_ch1=512, out_ch=256, in_ch2=256,in_ch3=256)#in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder4(in_ch1=256, out_ch=128, in_ch2=128,in_ch3=512)
        self.decoder2 = Decoder4(in_ch1=128, out_ch=64, in_ch2=64,in_ch3=256)
        self.decoder1 = Decoder4(in_ch1=64, out_ch=64, in_ch2=64,in_ch3=64)
        #self.deconv4 = nn.ConvTranspose2d(512, filters[3], 3, stride=2, padding=1, output_padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)#zhuanzhi
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        c, f = self.vit(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        #x = self.firstact(x)
        x = self.firstmaxpool(x_0)
        e1 = self.encoder1(x)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)



        g = self.gcm(e4)
        x_0, e1, e2, e3 = self.mlfc1(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc2(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc3(x_0, e1, e2, e3)
        # Decoder
        up4 = self.decoder4(e4,e3,g[0],e3)
        up3 = self.decoder3(up4,e2,g[1],f[0])
        up2 = self.decoder2(up3,e1,g[2],f[1])
        up1 = self.decoder1(up2,x_0,g[3],f[2]) # f[2]:(2, 1024, 38, 38)

        out = self.finaldeconv1(up1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class deform_up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch1, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nonlinearity
        #self.coatt = CCA(F_g=out_ch // 2, F_x= in_ch2 // 2)
        # self.ce = Context_Exploration_Block(in_ch2)
        self.conv = double_deform_conv(out_ch + in_ch2 + in_ch2, out_ch)
        #self.residual = Residual(out_ch + in_ch2 + in_ch2, out_ch)
        self.se = SEWeightModule(out_ch + in_ch2 + in_ch2)
        self.dff = DFF(in_ch2)
    def forward(self, x1, x2, x3):
        x1 = self.deconv(x1)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        #skip_x_att = self.coatt(g=x1, x=x2)
        # input is CHW
        # channel attetion
        # c2 = self.ce(x2)
        # c = torch.cat([x1, x2, x3], dim=1)
        output=self.dff(x1, x2, x3)
        # w = self.se(output)*output
        fuse = self.conv(output)
        return fuse
#+MLFC+snack
class UCRNet10_2(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2,n_filts = 64):
        super(UCRNet10_2, self).__init__()

        filters = [64, 128, 256, 512]
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = self.replace_conv_with_snakeconv(resnet.layer1)
        self.encoder2 = self.replace_conv_with_snakeconv(resnet.layer2)
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.gcm = GCM(512,64)
        self.mlfc1 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc2 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc3 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)

        #self.dblock = Dblock(512)
        self.decoder4 = Decoder2(in_ch1=512, out_ch=256, in_ch2=256)
        self.decoder3 = Decoder2(in_ch1=256, out_ch=128, in_ch2=128)
        self.decoder2 = deform_up(in_ch1=128, out_ch=64, in_ch2=64)
        self.decoder1 = deform_up(in_ch1=64, out_ch=64, in_ch2=64)
        #self.deconv4 = nn.ConvTranspose2d(512, filters[3], 3, stride=2, padding=1, output_padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)#zhuanzhi
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.drop = nn.Dropout2d(drop_rate)

    def replace_conv_with_snakeconv(self, block):
        for name, module in block.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(block, name, DeformConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias
                ))
            elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
                self.replace_conv_with_snakeconv(module)
        return block

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        #x = self.firstact(x)
        x = self.firstmaxpool(x_0)
        e1 = self.encoder1(x)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)



        g = self.gcm(e4)
        x_0, e1, e2, e3 = self.mlfc1(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc2(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc3(x_0, e1, e2, e3)
        # Decoder
        up4 = self.decoder4(e4,e3,g[0])
        up3 = self.decoder3(up4,e2,g[1])
        up2 = self.decoder2(up3,e1,g[2])
        up1 = self.decoder1(up2,x_0,g[3]) # f[2]:(2, 1024, 38, 38)

        out = self.finaldeconv1(up1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out
#+MLFC
class UCRNet10_1(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2,n_filts = 64):
        super(UCRNet10_1, self).__init__()

        filters = [64, 128, 256, 512]
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.gcm = GCM(512,64)
        self.mlfc1 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc2 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc3 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)

        #self.dblock = Dblock(512)
        self.decoder4 = Decoder2(in_ch1=512, out_ch=256, in_ch2=256)#in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder2(in_ch1=256, out_ch=128, in_ch2=128)
        self.decoder2 = Decoder2(in_ch1=128, out_ch=64, in_ch2=64)
        self.decoder1 = Decoder2(in_ch1=64, out_ch=64, in_ch2=64)
        #self.deconv4 = nn.ConvTranspose2d(512, filters[3], 3, stride=2, padding=1, output_padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)#zhuanzhi
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        #x = self.firstact(x)
        x = self.firstmaxpool(x_0)
        e1 = self.encoder1(x)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)



        g = self.gcm(e4)
        x_0, e1, e2, e3 = self.mlfc1(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc2(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc3(x_0, e1, e2, e3)
        # Decoder
        up4 = self.decoder4(e4,e3,g[0])
        up3 = self.decoder3(up4,e2,g[1])
        up2 = self.decoder2(up3,e1,g[2])
        up1 = self.decoder1(up2,x_0,g[3]) # f[2]:(2, 1024, 38, 38)

        out = self.finaldeconv1(up1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out
#+MLFC
class UCRNet10_1(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2,n_filts = 64):
        super(UCRNet10_1, self).__init__()

        filters = [64, 128, 256, 512]
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.gcm = GCM(512,64)
        self.mlfc1 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc2 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)
        self.mlfc3 = MLFC(n_filts, n_filts, n_filts * 2, n_filts * 4, lenn=1)

        #self.dblock = Dblock(512)
        self.decoder4 = Decoder2(in_ch1=512, out_ch=256, in_ch2=256)#in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder2(in_ch1=256, out_ch=128, in_ch2=128)
        self.decoder2 = Decoder2(in_ch1=128, out_ch=64, in_ch2=64)
        self.decoder1 = Decoder2(in_ch1=64, out_ch=64, in_ch2=64)
        #self.deconv4 = nn.ConvTranspose2d(512, filters[3], 3, stride=2, padding=1, output_padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)#zhuanzhi
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        #x = self.firstact(x)
        x = self.firstmaxpool(x_0)
        e1 = self.encoder1(x)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)



        g = self.gcm(e4)
        x_0, e1, e2, e3 = self.mlfc1(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc2(x_0, e1, e2, e3)
        x_0, e1, e2, e3 = self.mlfc3(x_0, e1, e2, e3)
        # Decoder
        up4 = self.decoder4(e4,e3,g[0])
        up3 = self.decoder3(up4,e2,g[1])
        up2 = self.decoder2(up3,e1,g[2])
        up1 = self.decoder1(up2,x_0,g[3]) # f[2]:(2, 1024, 38, 38)

        out = self.finaldeconv1(up1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class UCRNet10_1_or(nn.Module):#最原始的
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2):
        super(UCRNet10_1_or, self).__init__()

        filters = [64, 128, 256, 512]
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.gcm = GCM(512,64)
        #self.dblock = Dblock(512)
        self.decoder4 = Decoder4(in_ch1=512, out_ch=256, in_ch2=256)
        self.decoder3 = Decoder4(in_ch1=256, out_ch=128, in_ch2=128)
        self.decoder2 = Decoder4(in_ch1=128, out_ch=64, in_ch2=64)
        self.decoder1 = Decoder4(in_ch1=64, out_ch=64, in_ch2=64)
        #self.deconv4 = nn.ConvTranspose2d(512, filters[3], 3, stride=2, padding=1, output_padding=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        #x = self.firstact(x)
        x = self.firstmaxpool(x_0)
        e1 = self.encoder1(x)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        e4 = self.drop(e4)

        g = self.gcm(e4)

        # Decoder
        up4 = self.decoder4(e4,e3,g[0])
        up3 = self.decoder3(up4,e2,g[1])
        up2 = self.decoder2(up3,e1,g[2])
        up1 = self.decoder1(up2,x_0,g[3])

        out = self.finaldeconv1(up1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out
"进行decoder的封装"
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

###############################################################################################################################
class Decoder3(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0,in_ch3=0, attn=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch1, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nonlinearity
        #self.coatt = CCA(F_g=out_ch // 2, F_x= in_ch2 // 2)
        # self.ce = Context_Exploration_Block(in_ch2)
        self.conv = DoubleConv(out_ch + in_ch2 + in_ch2+in_ch3, out_ch)
        self.residual = Residual(out_ch+ in_ch2 + in_ch2+in_ch3+in_ch3, out_ch)
        self.se = SEWeightModule(out_ch + in_ch2 + in_ch2+ in_ch3)
        self.dff = DFF(in_ch2)
        '''
        self.decoder4 = Decoder4(in_ch1=512, out_ch=256, in_ch2=256,in_ch3=256)#in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder4(in_ch1=256, out_ch=128, in_ch2=128,in_ch3=512)
        self.decoder2 = Decoder4(in_ch1=128, out_ch=64, in_ch2=64,in_ch3=256)
        self.decoder1 = Decoder4(in_ch1=64, out_ch=64, in_ch2=64,in_ch3=64)
        '''


    def forward(self, x1, x2, x3,x4):  # e3:(2, 256, 38, 38) , e4:(2, 512, 19, 19) , f[2]:(2, 64, 304, 304) , g[0]:(2, 256, 38, 38)
        x1 = self.deconv(x1)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        #skip_x_att = self.coatt(g=x1, x=x2)
        # input is CHW
        # channel attetion
        # c2 = self.ce(x2)
        output = self.dff(x1, x2, x3)
        c = torch.cat([output, x4], dim=1)

        # w = self.se(c)*c
        #c = self.residual(w)
        fuse = self.conv(c)
        return fuse
class Decoder2(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch1, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nonlinearity
        #self.coatt = CCA(F_g=out_ch // 2, F_x= in_ch2 // 2)
        # self.ce = Context_Exploration_Block(in_ch2)
        self.conv = DoubleConv(out_ch + in_ch2 + in_ch2, out_ch)
        # self.residual = Residual(out_ch+ in_ch2 + in_ch2, out_ch)
        self.se = SEWeightModule(out_ch + in_ch2 + in_ch2)

        '''
        self.decoder4 = Decoder4(in_ch1=512, out_ch=256, in_ch2=256,in_ch3=256)#in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder4(in_ch1=256, out_ch=128, in_ch2=128,in_ch3=512)
        self.decoder2 = Decoder4(in_ch1=128, out_ch=64, in_ch2=64,in_ch3=256)
        self.decoder1 = Decoder4(in_ch1=64, out_ch=64, in_ch2=64,in_ch3=64)
        '''


    def forward(self, x1, x2, x3):  # e3:(2, 256, 38, 38) , e4:(2, 512, 19, 19) , f[2]:(2, 64, 304, 304) , g[0]:(2, 256, 38, 38)
        x1 = self.deconv(x1)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        #skip_x_att = self.coatt(g=x1, x=x2)
        # input is CHW
        # channel attetion
        # c2 = self.ce(x2)
        c = torch.cat([x1, x2, x3], dim=1)
        w = self.se(c)*c
        #c = self.residual(w)
        fuse = self.conv(w)

        return fuse
class Decoder4(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0,in_ch3=0, attn=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch1, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nonlinearity
        #self.coatt = CCA(F_g=out_ch // 2, F_x= in_ch2 // 2)
        # self.ce = Context_Exploration_Block(in_ch2)
        self.conv = DoubleConv(out_ch + in_ch2 + in_ch2+in_ch3, out_ch)
        self.residual = Residual(out_ch+ in_ch2 + in_ch2+in_ch3+in_ch3, out_ch)
        self.se = SEWeightModule(out_ch + in_ch2 + in_ch2+ in_ch3)

        '''
        self.decoder4 = Decoder4(in_ch1=512, out_ch=256, in_ch2=256,in_ch3=256)#in_ch1=512, out_ch=256, in_ch2=256
        self.decoder3 = Decoder4(in_ch1=256, out_ch=128, in_ch2=128,in_ch3=512)
        self.decoder2 = Decoder4(in_ch1=128, out_ch=64, in_ch2=64,in_ch3=256)
        self.decoder1 = Decoder4(in_ch1=64, out_ch=64, in_ch2=64,in_ch3=64)
        '''


    def forward(self, x1, x2, x3,x4):  # e3:(2, 256, 38, 38) , e4:(2, 512, 19, 19) , f[2]:(2, 64, 304, 304) , g[0]:(2, 256, 38, 38)
        x1 = self.deconv(x1)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        #skip_x_att = self.coatt(g=x1, x=x2)
        # input is CHW
        # channel attetion
        # c2 = self.ce(x2)
        c = torch.cat([x1, x2, x3, x4], dim=1)

        w = self.se(c)*c
        #c = self.residual(w)
        fuse = self.conv(w)

        return fuse
class Decoder1(nn.Module):#最原始的
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch1, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nonlinearity
        #self.coatt = CCA(F_g=out_ch // 2, F_x= in_ch2 // 2)
        self.ce = Context_Exploration_Block(in_ch2)
        self.conv = DoubleConv(out_ch + in_ch2 + in_ch2, out_ch)
        #self.residual = Residual(out_ch + in_ch2 + in_ch2, out_ch)
        self.se = SEWeightModule(out_ch + in_ch2 + in_ch2)

    def forward(self, x1, x2, x3):
        x1 = self.deconv(x1)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        #skip_x_att = self.coatt(g=x1, x=x2)
        # input is CHW
        # channel attetion
        c2 = self.ce(x2)
        c = torch.cat([x1, c2, x3], dim=1)
        w = self.se(c)*c
        fuse = self.conv(w)
        return fuse
###############################################################################################################################
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x





