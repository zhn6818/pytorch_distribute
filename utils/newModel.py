# _*_ coding: utf-8 _*_
"""
Time:     2020/11/22 19:06
Author:   Cheng Ding(Deeachain)
Version:  V 0.1
File:     FCN8s.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

######################################################################################
# FCNFCNnew: Fully Convolutional Networks for Semantic Segmentation
# Paper-Link: https://arxiv.org/abs/1411.4038
######################################################################################

__all__ = ["FCNnew"]


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv3x3_block_x1(nn.Module):
    '''(conv => BN => ReLU) * 1'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_block_x2(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_block_x3(nn.Module):
    '''(conv => BN => ReLU) * 3'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(upsample, self).__init__()
        self.conv1x1 = conv1x1(in_ch, out_ch)
        self.scale_factor = scale_factor

    def forward(self, H):
        """
        H: High level feature map, upsample
        """
        H = self.conv1x1(H)
        H = F.interpolate(H, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return H


class FCNnew(nn.Module):
    def __init__(self, num_classes):
        super(FCNnew, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = conv3x3_block_x2(3, 64)
        self.block2 = conv3x3_block_x2(64, 128)
        self.block3 = conv3x3_block_x3(128, 256)
        self.block4 = conv3x3_block_x3(256, 512)
        self.block5 = conv3x3_block_x3(512, 1024)
        self.block6 = conv3x3_block_x3(1024, 1024)
        self.upsample1 = upsample(1024, 1024, 2)
        self.conv1 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=False)
        # self.upsample2 = upsample(512, 256, 2)
        # self.upsample3 = upsample(256, num_classes, 8)

    def forward(self, x):
        block1_x = self.block1(x)
        block2_x = self.block2(block1_x)
        block3_x = self.block3(block2_x)
        block4_x = self.block4(block3_x)
        block5_x = self.block5(block4_x)
        block6_x = self.block6(block5_x)
        upsample1 = self.upsample1(block6_x)
        x = torch.add(upsample1, block5_x)
        x = self.conv1(x)
        # upsample2 = self.upsample2(x)
        # x = torch.add(upsample2, block3_x)
        # x = self.upsample3(x)

        return x


if __name__ == '__main__':
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    model = FCNnew(num_classes=2)
    summary(model, (3, 512, 512), device="cpu")
    input = torch.rand(1, 3, 512, 512).cuda()
    model.cuda()
    model.eval()
    output = model(input)
    print(output.shape)
    
    # summary(model, (3, 512, 512), device="cpu")
    
