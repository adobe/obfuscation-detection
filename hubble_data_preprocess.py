import torch
import pandas as pd
import random
import gc
random.seed(42)

TOTAL_SAMPLES = 32117072
NUM_SAMPLES = 50000
TENSOR_LENGTH = 4096
CHAR_DICT = torch.load('char_dict.pth')
print(CHAR_DICT)
hubble_csv = pd.read_csv('data/hubble_windows_wilson.csv')
print(hubble_csv.shape[0])

# choose random 100k samples from whole dataset
random_idx = random.sample(range(TOTAL_SAMPLES), NUM_SAMPLES)
print(hubble_csv.loc[0]['process'])
print(len(random_idx))
print(random_idx[0:5])

hubble_x = []
hubble_y = [torch.Tensor([1, 0]) for _ in range(NUM_SAMPLES)]
cmds = []
x = 0
for idx in random_idx:
    if x % 10000 == 0:
        print(x)
    x += 1

    cmd = hubble_csv.loc[idx]['process']
    cmd_tensor = torch.zeros(len(CHAR_DICT) + 1, TENSOR_LENGTH)
    tensor_len = min(TENSOR_LENGTH, len(cmd))

    for i in range(tensor_len):
        char = cmd[i]
        lower_char = char.lower()
        if char.isupper() and lower_char in CHAR_DICT:
            cmd_tensor[len(CHAR_DICT)][i] = 1
            char = lower_char
        if char in CHAR_DICT:
            cmd_tensor[CHAR_DICT[char]][i] = 1
    
    hubble_x.append(cmd_tensor)
    cmds.append(cmd)

# free memory from csv
del hubble_csv
gc.collect()

hubble_x = torch.stack(hubble_x)
hubble_y = torch.stack(hubble_y)

print(cmds[0])
print(hubble_x[0][46][0])
print(hubble_x[0][73][0])
print(hubble_x[0][13][6])
print(hubble_x[0][73][6])
print(hubble_y[0])
print(hubble_x.shape)
print(hubble_y.shape)

torch.save({'x': hubble_x, 'y': hubble_y}, 'data/hubble_data.pth')
torch.save(cmds, 'hubble_cmds.pth')
