import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

def conv3x3_leakyrelu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True),
        nn.LeakyReLU(0.1, inplace = True))

def conv1x1_leakyrelu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            bias=True),
        nn.LeakyReLU(0.1, inplace = True))

def depth_layer(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, padding=1, stride=1, bias=True)

class DepthNet(nn.Module):

    def __init__(self, info_size):
        super(DepthNet, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv0 = conv3x3_leakyrelu(3, 16, stride=1)
        self.conv1_0 = conv3x3_leakyrelu(16, 32, stride=2)

        self.info_layer = conv1x1_leakyrelu(info_size, 32)

        self.conv1_1 = conv3x3_leakyrelu(32+2+8+32, 64, stride=1)
        self.conv1_2 = conv3x3_leakyrelu(64, 64, stride=1)
        self.conv1_3 = conv3x3_leakyrelu(64, 64, stride=1)

        self.conv2_0 = conv3x3_leakyrelu(64, 128, stride=2)
        self.conv2_1 = conv3x3_leakyrelu(128, 128, stride=1)
        self.conv2_2 = conv3x3_leakyrelu(128, 128, stride=1)

        self.conv3_0 = conv3x3_leakyrelu(128, 256, stride=2)
        self.conv3_1 = conv3x3_leakyrelu(256, 256, stride=1)
        self.conv3_2 = conv3x3_leakyrelu(256, 256, stride=1)

        self.conv4_0 = conv3x3_leakyrelu(256, 512, stride=2)
        self.conv4_1 = conv3x3_leakyrelu(512, 512, stride=1)
        self.conv4_2 = conv3x3_leakyrelu(512, 512, stride=1)

        self.conv5_0 = conv3x3_leakyrelu(512, 512, stride=2)
        self.conv5_1 = conv3x3_leakyrelu(512, 512, stride=1)
        self.conv5_2 = conv3x3_leakyrelu(512, 512, stride=1)

        # decoder
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.delayer4_0 = conv3x3_leakyrelu(512 + 512, 512)
        self.delayer4_1 = conv3x3_leakyrelu(512, 512)
        self.delayer4_2 = conv3x3_leakyrelu(512, 512)

        self.delayer3_0 = conv3x3_leakyrelu(512 + 256, 256)
        self.delayer3_1 = conv3x3_leakyrelu(256, 256)
        self.delayer3_2 = conv3x3_leakyrelu(256, 256)
        self.depth3 = depth_layer(256)

        self.delayer2_0 = conv3x3_leakyrelu(256 + 128 + 1, 128)
        self.delayer2_1 = conv3x3_leakyrelu(128, 128)
        self.delayer2_2 = conv3x3_leakyrelu(128, 128)
        self.depth2 = depth_layer(128)
        
        self.delayer1_0 = conv3x3_leakyrelu(128 + 64 + 1, 64)
        self.delayer1_1 = conv3x3_leakyrelu(64, 64)
        self.delayer1_2 = conv3x3_leakyrelu(64, 64)
        self.depth1 = depth_layer(64)

        # the following is used to normalize the triangulation layer before any process
        tri_mean = [79.589,63.723,0.993,4.738,6.135,0.088,80.223,63.629]
        self.tri_mean = torch.tensor(tri_mean).cuda().view(1, 8, 1, 1)
        tri_std = [4.775e+01,3.840e+01,3.637e-02,9.760e+01,8.440e+01,5.954e-01,4.806e+01,3.914e+01]
        self.tri_std = torch.tensor(tri_std).cuda().view(1, 8, 1, 1)
        flow_mean = [0.476, 0.052]
        self.flow_mean = torch.tensor(flow_mean).cuda().view(1, 2, 1, 1)
        flow_std = [15.028, 13.412]
        self.flow_std = torch.tensor(flow_std).cuda().view(1, 2, 1, 1)

    def forward(self, img, info, flow, triangle):
        flow = (flow - self.flow_mean)/ self.flow_std
        triangle = (triangle - self.tri_mean)/ self.tri_std

        l0 = self.conv0(img)
        l1 = self.conv1_0(l0)

        info = self.info_layer(info)
        enl1 = torch.cat([l1, info, flow, triangle], dim = 1)
        enl1 = self.conv1_3(self.conv1_2(self.conv1_1(enl1)))
        enl2 = self.conv2_2(self.conv2_1(self.conv2_0(enl1)))
        enl3 = self.conv3_2(self.conv3_1(self.conv3_0(enl2)))
        enl4 = self.conv4_2(self.conv4_1(self.conv4_0(enl3)))
        enl5 = self.conv5_2(self.conv5_1(self.conv5_0(enl4)))

        del4 = torch.cat([self.upsample(enl5), enl4], dim =1)
        del4 = self.delayer4_2(self.delayer4_1(self.delayer4_0(del4)))
        
        del3 = torch.cat([self.upsample(del4), enl3], dim =1)
        del3 = self.delayer3_2(self.delayer3_1(self.delayer3_0(del3)))
        depth3 = self.depth3(del3)
        
        del2 = torch.cat([self.upsample(del3), enl2, self.upsample(depth3)], dim =1)
        del2 = self.delayer2_2(self.delayer2_1(self.delayer2_0(del2)))
        depth2 = self.depth2(del2)

        del1 = torch.cat([self.upsample(del2), enl1, self.upsample(depth2)], dim =1)
        del1 = self.delayer1_2(self.delayer1_1(self.delayer1_0(del1)))
        depth1 = self.depth1(del1)

        return [depth1, depth2, depth3]