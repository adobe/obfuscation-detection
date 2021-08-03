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
CSV_DIR = 'processed_csv/'
TOTAL_SAMPLES = 98123
NUM_SAMPLES = 41402
# substrings that we found contained obfuscated code
OBFUSCATED = torch.load(DATA_DIR + 'obfuscated-str.pth')
# OBFUSCATED = []
print('num obfuscated:', len(OBFUSCATED))
cb_csv = pd.read_csv(DATA_DIR + 'win_cmds_cb.csv')
print(cb_csv.shape[0])

# choose random 100k samples from whole dataset
random_idx = random.sample(range(TOTAL_SAMPLES), NUM_SAMPLES)
print(cb_csv.loc[0]['process'])
print(len(random_idx))
print(random_idx[0:5])

dataset = []
x = 0
num_pos = 0
for i in range(len(random_idx)):
    if x % 10000 == 0:
        print(x)
    x += 1

    cmd = cb_csv.loc[random_idx[i]]['process']
    
    # label y
    is_obf = 0
    for obf in OBFUSCATED:
        if obf in cmd:
            is_obf = 1
            num_pos += 1
            break
    
    dataset.append([is_obf, cmd])

df = pd.DataFrame(dataset, columns=['label', 'command'])
df.to_csv(DATA_DIR + CSV_DIR + 'cb-data.csv', index=False)

# sanity checks
print('num pos:', num_pos)
print(len(df))
