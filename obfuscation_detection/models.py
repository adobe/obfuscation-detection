# 
# Authors: Security Intelligence Team within the Security Coordination Center
# 
# Copyright 2021 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS 
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#

import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = self.tanh(x[:, :int(x.shape[1]/2)])
        s = self.sigmoid(x[:, int(x.shape[1]/2):])
        return t * s

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearNorm, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain('linear'))
    
    def forward(self, x):
        x = self.flatten(x) # flatten b/c it's too large for fc right now?
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        nn.init.xavier_normal_(
            self.conv.weight,
            gain=nn.init.calculate_gain('tanh'))

    def forward(self, x):
        return self.conv(x)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        NUM_LAYERS = 3
        NUM_FILTERS = 512
        input_size = 74
        convolutions_char = []
        self.num_filters = NUM_FILTERS
        for _ in range(NUM_LAYERS):
            conv_layer = nn.Sequential(
                ConvNorm(input_size, NUM_FILTERS,
                            kernel_size=5, stride=1,
                            padding=2, dilation=1),
                nn.BatchNorm1d(NUM_FILTERS)
            )
            convolutions_char.append(conv_layer)
            input_size = NUM_FILTERS // 2
        self.convolutions_char = nn.ModuleList(convolutions_char)
        self.pre_out = LinearNorm(NUM_FILTERS // 2, 2)
    
    def forward(self, x):
        # x = x.permute(0, 2, 1)
        half = self.num_filters // 2
        res = None
        skip = None
        for i in range(len(self.convolutions_char)):
            conv = self.convolutions_char[i]
            drop = True
            if i >= len(self.convolutions_char) - 1:
                drop = False
            if skip is not None:
                x = x + skip
            conv_out = conv(x)
            tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid(conv_out[:, half:, :])
            if res is None:
                res = tmp
            else:
                res = res + tmp
            skip = tmp
            x = torch.dropout(tmp, 0.1, drop)
        x = x + res
        # x = x.permute(0, 2, 1)
        pre = torch.sum(x, dim=2, dtype=torch.float)
        pre /= 4096
        return torch.softmax(self.pre_out(pre), dim=1)
