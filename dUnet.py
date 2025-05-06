#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)

        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1) 
        x = self.conv(x)

        return x


class model1(nn.Module):
    def __init__(self, channels, input_channels):
        super(model1, self).__init__()

        """ Encoder """
        self.e1 = encoder_block(input_channels, channels)
        self.e2 = encoder_block(channels, channels)
        self.e3 = encoder_block(channels, channels)

        """ Bottleneck """
        self.b = conv_block(channels, channels)

        """ Decoder """
        self.d1 = decoder_block(channels, channels)
        self.d2 = decoder_block(channels, channels)
        self.d3 = decoder_block(channels, channels)
      
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        """ Bottleneck """
        b = self.b(p3)

        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        
        return d3
        
class model2(nn.Module):
    def __init__(self, channels, input_channels):
        super(model2,self).__init__()

        """ Encoder """
        self.e1 = encoder_block(input_channels, channels)
        self.e2 = encoder_block(channels, channels)
        self.e3 = encoder_block(channels, channels)

        """ Bottleneck """
        self.b = conv_block(channels, channels)

        """ Decoder """
        self.d1 = decoder_block(channels, channels)
        self.d2 = decoder_block(channels, channels)
        self.d3 = decoder_block(channels, channels)
        
    def forward(self, inputs):
        
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        """ Bottleneck """
        b = self.b(p3)

        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        
        return(d3)
    
class doubleUnet(nn.Module):
    def __init__(self, model1, model2, channels, output_channels):
        super(doubleUnet, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
        #concat outputs from model 1 and model 2
        self.conv = conv_block(2*channels, channels)
        self.outputs = nn.Conv3d(channels, output_channels, kernel_size=1, padding=0)


    def forward(self, x1,x2):
        x1 = self. model1(x1)
        x2 = self. model2(x2)
        
        x = torch.cat([x1, x2], axis=1)
        x = self.conv(x)
        x = self.outputs(x)
        return x