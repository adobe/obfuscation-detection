import torch
import pandas as pd
import random
random.seed(42)

DATA_DIR = '../data/'
TOTAL_SAMPLES = 20480042
NUM_SAMPLES = 17180
TENSOR_LENGTH = 4096
# indices only work for current csv file and current 65180 NUM_SAMPLES
# OBFUSCATED = [4472, 7525, 9402, 10990, 12029, 19751, 28577, 32235, 44386, 47108, 49523,\
#                 56473, 57113, 58178, 60861, 17273, 36984, 59022]
OBFUSCATED = []
print('num obfuscated:', len(OBFUSCATED))
CHAR_DICT = torch.load(DATA_DIR + 'prep/char_dict.pth')
print(CHAR_DICT)
hubble_csv = pd.read_csv(DATA_DIR + 'win_cmds_hubble.csv')
print(hubble_csv.shape[0])

# choose random 100k samples from whole dataset
random_idx = random.sample(range(TOTAL_SAMPLES), NUM_SAMPLES)
print(hubble_csv.loc[0]['process'])
print(len(random_idx))
print(random_idx[0:5])

hubble_x = torch.zeros(NUM_SAMPLES, len(CHAR_DICT) + 1, TENSOR_LENGTH, dtype=torch.int8)
hubble_y = torch.stack([torch.tensor([1, 0], dtype=torch.int8) for _ in range(NUM_SAMPLES)])
cmds = []
x = 0
for i in range(len(random_idx)):
    if x % 10000 == 0:
        print(x)
    x += 1

    cmd = hubble_csv.loc[random_idx[i]]['process']
    tensor_len = min(TENSOR_LENGTH, len(cmd))

    for j in range(tensor_len):
        char = cmd[j]
        lower_char = char.lower()
        if char.isupper() and lower_char in CHAR_DICT:
            hubble_x[i][len(CHAR_DICT)][j] = 1
            char = lower_char
        if char in CHAR_DICT:
            hubble_x[i][CHAR_DICT[char]][j] = 1
    
    cmds.append(cmd)

for o in OBFUSCATED:
    print(cmds[o])
    hubble_y[o][0] = 0
    hubble_y[o][1] = 1

print(cmds[0])
print(hubble_x[0][23][0])
print(hubble_x[0][73][0])
print(hubble_x[0][67][2])
print(hubble_x[0][73][2])
print(hubble_y[0])
# print(hubble_y[OBFUSCATED[2]])
print(hubble_x.shape, hubble_x.dtype)
print(hubble_y.shape, hubble_y.dtype)

torch.save({'x': hubble_x, 'y': hubble_y}, DATA_DIR + 'processed_tensors/hubble_data.pth')
torch.save(cmds, DATA_DIR + 'scripts/hubble_cmds.pth')
