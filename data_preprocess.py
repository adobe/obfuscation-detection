import torch

DATA_DIR = 'data/processed_tensors/'
ps_tensors = torch.load(DATA_DIR + 'ps_data.pth')
print('loaded ps data:', ps_tensors['x'].shape, ps_tensors['y'].shape)
dos_tensors = torch.load(DATA_DIR + 'dos_data.pth')
print('loaded dos data:', dos_tensors['x'].shape, dos_tensors['y'].shape)
hubble_tensors = torch.load(DATA_DIR + 'hubble_data.pth')
print('loaded hubble data:', hubble_tensors['x'].shape, hubble_tensors['y'].shape)

ps_scripts = torch.load('ps_scripts.pth')
dos_cmds = torch.load('dos_cmds.pth')
hubble_cmds = torch.load('hubble_cmds.pth')

all_tensors_x = torch.cat((ps_tensors['x'], dos_tensors['x'], hubble_tensors['x']))
all_tensors_y = torch.cat((ps_tensors['y'], dos_tensors['y'], hubble_tensors['y']))
all_scripts = ps_scripts + dos_cmds + hubble_cmds
# all_tensors_x = torch.cat((dos_tensors['x'], hubble_tensors['x']))
# all_tensors_y = torch.cat((dos_tensors['y'], hubble_tensors['y']))
# all_scripts = dos_cmds + hubble_cmds
print('all tensors:', all_tensors_x.shape, all_tensors_y.shape)
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
                                                            [25607, 4801, 1600], # 32008 split
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
torch.save(train_cmds, 'train_cmds_list.pth')
torch.save(val_cmds, 'val_cmds_list.pth')
torch.save(test_cmds, 'test_cmds_list.pth')