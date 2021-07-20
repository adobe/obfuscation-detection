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

all_tensors = torch.cat((ps_tensors, dos_tensors, hubble_tensors))
all_scripts = ps_scripts + dos_cmds + hubble_cmds
print('all tensors:', all_tensors.shape)
print('scripts:', len(all_scripts))

# train-dev-test split of 80-15-5
