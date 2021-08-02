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
TENSOR_LENGTH = 4096
CHAR_DICT = torch.load(DATA_DIR + 'prep/char_dict.pth')
print(CHAR_DICT)
obf_csv = pd.read_csv(DATA_DIR + 'new-positives.csv', sep='\t')
NUM_SAMPLES = obf_csv.shape[0]
print(obf_csv.shape[0])

obf_x = torch.zeros(NUM_SAMPLES, len(CHAR_DICT) + 1, TENSOR_LENGTH, dtype=torch.int8)
obf_y = torch.stack([torch.tensor([0, 1], dtype=torch.int8) for _ in range(NUM_SAMPLES)])
cmds = []
x = 0
num_pos = 0
for i in range(NUM_SAMPLES):
    if x % 10000 == 0:
        print(x)
    x += 1

    cmd = obf_csv.loc[i]['process']
    tensor_len = min(TENSOR_LENGTH, len(cmd))

    # label x
    for j in range(tensor_len):
        char = cmd[j]
        lower_char = char.lower()
        if char.isupper() and lower_char in CHAR_DICT:
            obf_x[i][len(CHAR_DICT)][j] = 1
            char = lower_char
        if char in CHAR_DICT:
            obf_x[i][CHAR_DICT[char]][j] = 1

    cmds.append(cmd)

print(cmds[0])
print(obf_x[0][23][1])
print(obf_x[0][73][1])
print(obf_x[0][67][3])
print(obf_x[0][73][3])
print(obf_y[0])
print(obf_x[629][59][0])
print(obf_x[629][73][0])
print(obf_y[629])
# print(obf_y[OBFUSCATED[2]])
print(obf_x.shape, obf_x.dtype)
print(obf_y.shape, obf_y.dtype)

torch.save({'x': obf_x, 'y': obf_y}, DATA_DIR + 'processed_tensors/obf_data.pth')
torch.save(cmds, DATA_DIR + 'scripts/obf_cmds.pth')
