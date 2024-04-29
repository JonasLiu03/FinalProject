import torch
from torch import nn
import torch.nn.functional as F
import math


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
class AttentionBlock(nn.Module):
    def __init__(self, n_channels):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(n_channels, n_channels // 16, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_channels // 16, n_channels, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_depthwise=False, batch_norm=False, use_group_norm=False, activation=None, dropout_rate=0.0, dilation=1):
        super(ConvolutionalBlock, self).__init__()
        layers = []
        padding = (kernel_size // 2) * dilation
        if use_depthwise:
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation))
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        elif use_group_norm:
            num_groups = 32
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        if activation is not None:
            activation_layer = {
                'prelu': nn.PReLU(),
                'mish': Mish()
            }.get(activation.lower())
            if activation_layer:
                layers.append(activation_layer)
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        self.conv_block = nn.Sequential(*layers)
    def forward(self, input):
        return self.conv_block(input)

class SubPixelConvolutionalBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2, use_depthwise=False, dilation=1):
        super(SubPixelConvolutionalBlock, self).__init__()

        if use_depthwise:
            self.conv = ConvolutionalBlock(in_channels=n_channels,
                                           out_channels=n_channels,
                                           kernel_size=kernel_size, use_depthwise=True, batch_norm=False, activation='prelu', dilation=dilation)
            self.conv1x1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2), kernel_size=1)
        else:
            self.conv = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                                           kernel_size=kernel_size, batch_norm=False, activation='prelu', dilation=dilation)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        if hasattr(self, 'conv1x1'):
            output = self.conv1x1(output)
        output = self.pixel_shuffle(output)
        output = self.prelu(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, activation='prelu', dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=activation, dilation=dilation)
        self.attention = AttentionBlock(n_channels)
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None, dilation=dilation)

    def forward(self, input):
        residual = input
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = output + residual
        return output
# class ResidualBlock(nn.Module):
#     def __init__(self, kernel_size=3, n_channels=64, activation='prelu', dilation=1):
#         super(ResidualBlock, self).__init__()
#         self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
#                                               batch_norm=True, activation=activation, dilation=dilation)
#         self.attention = AttentionBlock(n_channels)
#         self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
#                                               batch_norm=True, activation=None, dilation=dilation)
#
#     def forward(self, input):
#         residual = input
#         output = self.conv_block1(input)
#         attention_output = self.attention(output)
#         output = self.conv_block2(attention_output)
#         output = output + residual
#         return output, attention_output

    #Adjust!
class SRResNet(nn.Module):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=8, scaling_factor=4, dilation=1):
        super(SRResNet, self).__init__()
        self.conv_block1 = ConvolutionalBlock(3, n_channels, large_kernel_size, activation='prelu', init_type='kaiming', dilation=dilation)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels, dilation=dilation) for _ in range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(n_channels, n_channels, small_kernel_size, batch_norm=True, activation=None, dilation=dilation)
        self.subpixel_convolutional_blocks = nn.Sequential(*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2, dilation=dilation) for _ in range(int(math.log2(scaling_factor)))])
        self.conv_block3 = ConvolutionalBlock(n_channels, 3, large_kernel_size, batch_norm=False, activation='Tanh', dilation=dilation)

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block3(output)
        return sr_imgs
# class SRResNet(nn.Module):
#     def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=8, scaling_factor=4, dilation=1):
#         super(SRResNet, self).__init__()
#         self.conv_block1 = ConvolutionalBlock(3, n_channels, large_kernel_size, activation='prelu', init_type='kaiming', dilation=dilation)
#         self.residual_blocks = nn.Sequential(*[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels, dilation=dilation) for _ in range(n_blocks)])
#         self.conv_block2 = ConvolutionalBlock(n_channels, n_channels, small_kernel_size, batch_norm=True, activation=None, dilation=dilation)
#         self.subpixel_convolutional_blocks = nn.Sequential(*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2, dilation=dilation) for _ in range(int(math.log2(scaling_factor)))])
#         self.conv_block3 = ConvolutionalBlock(n_channels, 3, large_kernel_size, batch_norm=False, activation='Tanh', dilation=dilation)
#
#     def forward(self, lr_imgs):
#         output = self.conv_block1(lr_imgs)
#         residual = output
#         attention_maps = []
#
#         for block in self.residual_blocks:
#             output, attention_output = block(output)
#             attention_maps.append(attention_output.squeeze().cpu().detach().numpy())  # 存储每个残差块的注意力图
#
#         output = self.conv_block2(output)
#         output = output + residual
#         output = self.subpixel_convolutional_blocks(output)
#         sr_imgs = self.conv_block3(output)
#         return sr_imgs, attention_maps


