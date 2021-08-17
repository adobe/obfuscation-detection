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
from enum import Enum
from typing import List
import pkg_resources

from obfuscation_detection.models import ResNet

class PlatformType(Enum):
    WINDOWS: str='windows'
    LINUX: str='linux'
    ALL: str='all'

class ObfuscationClassifier:
    def __init__(self, platform: PlatformType):
        # set platform type
        if platform == PlatformType.WINDOWS:
            model_file = pkg_resources.resource_filename(__name__, 'files/best-resnet-windows.pth')
        elif platform == PlatformType.LINUX:
            model_file = pkg_resources.resource_filename(__name__, 'files/best-resnet-linux.pth')
        elif platform == PlatformType.ALL:
            model_file = pkg_resources.resource_filename(__name__, 'files/best-resnet-all.pth')
        else:
            raise Exception("Unknown platform type")
        
        # init device, model
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.model = ResNet()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.char_dict = torch.load(pkg_resources.resource_filename(__name__, 'files/char_dict.pth'))

        # load model
        checkpoint = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
    
    def __call__(self, commands_list: List[str]):
        BATCH_SIZE = 64
        indices = list(range(0, len(commands_list), BATCH_SIZE))
        for i in range(len(indices)):
            # split commands into batches
            if i < len(indices) - 1:
                commands = commands_list[i:i+1]
            else:
                commands = commands_list[i:]
            x = self._convert_batch(commands)
            y = self.model(x)
            # return 1 for obfuscated, 0 o.w.
            return list(torch.max(y, dim=1).indices.cpu().numpy())
    
    def _convert_batch(self, commands_list):
        TENSOR_LENGTH = 4096
        # convert batch of command strings into tensors
        x = torch.zeros(len(commands_list), len(self.char_dict) + 1, TENSOR_LENGTH).to(self.device)
        for i in range(len(commands_list)):
            cmd = commands_list[i]
            tensor_len = min(TENSOR_LENGTH, len(cmd))
            for j in range(tensor_len):
                char = cmd[j]
                lower_char = char.lower()
                if char.isupper() and lower_char in self.char_dict:
                    x[i][len(self.char_dict)][j] = 1
                    char = lower_char
                if char in self.char_dict:
                    x[i][self.char_dict[char]][j] = 1
        return x
