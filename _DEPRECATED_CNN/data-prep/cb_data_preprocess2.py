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
import pandas as pd
import random
random.seed(42)

DATA_DIR = '../data/'
TOTAL_SAMPLES = 100000
NUM_SAMPLES = 50000
TENSOR_LENGTH = 4096
# substrings that we found contained obfuscated code
OBFUSCATED = torch.load(DATA_DIR + 'obfuscated-str.pth')
print('num obfuscated:', len(OBFUSCATED))
CHAR_DICT = torch.load(DATA_DIR + 'prep/char_dict.pth')
print(CHAR_DICT)
cb_csv = pd.read_csv(DATA_DIR + 'win_cmds_cb_2.csv')
print(cb_csv.shape[0])

'''
# choose random 100k samples from whole dataset
random_idx = random.sample(range(TOTAL_SAMPLES), NUM_SAMPLES)
print(cb_csv.loc[0]['cmd'])
print(len(random_idx))
print(random_idx[0:5])
'''

# choose first or second 50k in the samples
random_idx = range(NUM_SAMPLES)
# random_idx = range(NUM_SAMPLES, TOTAL_SAMPLES)

cb_x = torch.zeros(NUM_SAMPLES, len(CHAR_DICT) + 1, TENSOR_LENGTH, dtype=torch.int8)
# cb_x = [[[0 for _ in range(TENSOR_LENGTH)] for _ in range(len(CHAR_DICT) + 1)] for _ in range(NUM_SAMPLES)]
print('loaded x')
cb_y = torch.stack([torch.tensor([1, 0], dtype=torch.int8) for _ in range(NUM_SAMPLES)])
# cb_y = [[1, 0] for _ in range(NUM_SAMPLES)]
print('loaded y')
cmds = []
x = 0
num_pos = 0
for i in range(len(random_idx)):
    if x % 10000 == 0:
        print(x)
    x += 1

    cmd = str(cb_csv.loc[random_idx[i]]['cmd'])
    tensor_len = min(TENSOR_LENGTH, len(cmd))

    # label x
    for j in range(tensor_len):
        char = cmd[j]
        lower_char = char.lower()
        if char.isupper() and lower_char in CHAR_DICT:
            cb_x[i][len(CHAR_DICT)][j] = 1
            char = lower_char
        if char in CHAR_DICT:
            cb_x[i][CHAR_DICT[char]][j] = 1
    
    # label y
    for obf in OBFUSCATED:
        if obf in cmd:
            cb_y[i][0] = 0
            cb_y[i][1] = 1
            num_pos += 1

    cmds.append(cmd)

print(cmds[0])
print(cb_x[0][59][0])
print(cb_x[0][73][0])
print(cb_x[0][23][1])
print(cb_x[0][73][1])
print(cb_y[0])
print(cb_x[1041][36][3])
print(cb_x[1041][73][3])
print(cb_y[1041])
print('num pos:', num_pos)
# print(cb_y[OBFUSCATED[2]])
print(cb_x.shape, cb_x.dtype)
print(cb_y.shape, cb_y.dtype)

# torch.save(cb_x, DATA_DIR + 'processed_tensors/cb2_x.pth')
# torch.save(cb_y, DATA_DIR + 'processed_tensors/cb2_y.pth')
torch.save({'x': cb_x, 'y': cb_y}, DATA_DIR + 'processed_tensors/cb2-1_data.pth')
torch.save(cmds, DATA_DIR + 'scripts/cb2-1_cmds.pth')
