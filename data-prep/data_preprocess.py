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

DATA_DIR = '../data/processed_tensors/'
SCRIPTS_DIR = '../data/scripts/'

all_tensors_x = []
all_tensors_y = []
all_scripts = []

all_tensors_x += list(torch.load(DATA_DIR + 'ps_data.pth')['x'])
all_tensors_y += list(torch.load(DATA_DIR + 'ps_data.pth')['y'])
print('loaded ps data', len(all_tensors_x))
all_tensors_x += list(torch.load(DATA_DIR + 'dos_data.pth')['x'])
all_tensors_y += list(torch.load(DATA_DIR + 'dos_data.pth')['y'])
print('loaded dos data', len(all_tensors_x))
all_tensors_x += list(torch.load(DATA_DIR + 'hubble_data.pth')['x'])
all_tensors_y += list(torch.load(DATA_DIR + 'hubble_data.pth')['y'])
print('loaded hubble data', len(all_tensors_x))
all_tensors_x += list(torch.load(DATA_DIR + 'cb_data.pth')['x'])
all_tensors_y += list(torch.load(DATA_DIR + 'cb_data.pth')['y'])
print('loaded cb data', len(all_tensors_x))
all_tensors_x += list(torch.load(DATA_DIR + 'obf_data.pth')['x'])
all_tensors_y += list(torch.load(DATA_DIR + 'obf_data.pth')['y'])
print('loaded obf data', len(all_tensors_x))

all_scripts += torch.load(SCRIPTS_DIR + 'ps_scripts.pth')
all_scripts += torch.load(SCRIPTS_DIR + 'dos_cmds.pth')
all_scripts += torch.load(SCRIPTS_DIR + 'hubble_cmds.pth')
all_scripts += torch.load(SCRIPTS_DIR + 'cb_cmds.pth')
all_scripts += torch.load(SCRIPTS_DIR + 'obf_cmds.pth')

print('all tensors:', len(all_tensors_x), len(all_tensors_y))
print('scripts:', len(all_scripts))

# # for unk_word_ratio
# torch.save(all_scripts, 'all_scripts.pth')
# print('saved all scripts')

# train-dev-test split of 80-15-5
train_cmds = []
val_cmds = []
test_cmds = []

# split all data into train, val, test
train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []
train_idx, val_idx, test_idx = torch.utils.data.random_split(range(len(all_tensors_x)), 
                                                            # hard-calculated 80-15-5 split
                                                            # [64000, 12000, 4000], # 80000 samples
                                                            # [32000, 6000, 2000], # 40000 samples
                                                            # [25607, 4801, 1600], # 32008 split
                                                            [40000, 7500, 2500], # 50000 samples
                                                            generator=torch.Generator().manual_seed(42))
for i in train_idx:
    train_x.append(all_tensors_x[i])
    train_y.append(all_tensors_y[i])
    train_cmds.append(all_scripts[i])
for i in val_idx:
    val_x.append(all_tensors_x[i])
    val_y.append(all_tensors_y[i])
    val_cmds.append(all_scripts[i])
for i in test_idx:
    test_x.append(all_tensors_x[i])
    test_y.append(all_tensors_y[i])
    test_cmds.append(all_scripts[i])
# free memory
all_tensors_x = None
all_tensors_y = None
all_scripts = None
# convert into tensor
train_x = torch.stack(train_x)
train_y = torch.stack(train_y)
val_x = torch.stack(val_x)
val_y = torch.stack(val_y)
test_x = torch.stack(test_x)
test_y = torch.stack(test_y)

print(train_x.shape, train_y.shape, train_x.dtype, train_y.dtype)
print(val_x.shape, val_y.shape, val_x.dtype, val_y.dtype)
print(test_x.shape, test_y.shape, test_x.dtype, test_y.dtype)

torch.save({'x': train_x, 'y': train_y}, DATA_DIR + 'train_data.pth')
torch.save({'x': val_x, 'y': val_y}, DATA_DIR + 'val_data.pth')
torch.save({'x': test_x, 'y': test_y}, DATA_DIR + 'test_data.pth')
torch.save(train_cmds, SCRIPTS_DIR + 'train_cmds_list.pth')
torch.save(val_cmds, SCRIPTS_DIR + 'val_cmds_list.pth')
torch.save(test_cmds, SCRIPTS_DIR + 'test_cmds_list.pth')
