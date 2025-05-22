import subprocess
import pandas as pd
import random
random.seed(42)

DATA_DIR = '../data/'
CSV_DIR = 'processed_csv/'
# install bashfuscator from https://github.com/Bashfuscator/Bashfuscator
BASHFUSCATOR_EXEC = '../../Bashfuscator/bashfuscator/bin/bashfuscator'
BASHFUSCATOR_TEMPLATE_SIMPLE = [BASHFUSCATOR_EXEC, '-q', '-c', '', '-s', '1', '-t', '1']
BASHFUSCATOR_TEMPLATE = [BASHFUSCATOR_EXEC, '-q', '-c',\
        '',\
        '-s', '1', '-t', '1',\
        '--choose-mutators']
MUTATORS = ['command/case_swapper', 'command/reverse', 'string/file_glob',\
            'string/folder_glob', 'string/hex_hash', 'token/forcode',\
            'token/special_char_only', 'encode/base64', 'encode/rotn',\
            'encode/xor_non_null', 'compress/bzip2', 'compress/gzip']

LINUX_CMDS = pd.read_csv(DATA_DIR + 'linux_cmds.csv')
NUM_CMDS = LINUX_CMDS.shape[0]
print(NUM_CMDS)

dataset = []

### add raw samples to dataset
for i in range(NUM_CMDS):
    dataset.append([0, 'none', LINUX_CMDS.loc[i]['command']])
print(len(dataset))
print('done raw samples')


### add bashfuscated samples to dataset
def bashfuscate(cmd, mutator_list):
    bashfuscator_cmd = list(BASHFUSCATOR_TEMPLATE)
    bashfuscator_cmd[3] = cmd
    for mutator in mutator_list:
        bashfuscator_cmd.append(mutator)
    return subprocess.run(bashfuscator_cmd, stdout=subprocess.PIPE, text=True)

# cmd = list(BASHFUSCATOR_TEMPLATE_SIMPLE)
# cmd[3] = 'cat /etc/passwd'
# # cmd.append('command/case_swapper')
# res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
# print(res)
# print(len(res.stdout))

# for i in plain_idx:
#     if i % 50 == 0:
#         print(i)
#     cmd = LINUX_CMDS.loc[i]['command']
#     bashfuscator_cmd = list(BASHFUSCATOR_TEMPLATE_SIMPLE)
#     bashfuscator_cmd[3] = cmd
#     bashfuscated = subprocess.run(bashfuscator_cmd, stdout=subprocess.PIPE, text=True)
#     if bashfuscated.returncode != 0:
#         print(cmd)
#     dataset.append([1, 'random', bashfuscated.stdout])
# print('done plain')

unparseable = 0

random_1_layer_idx = random.sample(range(NUM_CMDS), 10000)
print(random_1_layer_idx[0:5])
x = 0
for i in random_1_layer_idx:
    if x % 1000 == 0:
        print(x)
    x += 1

    try:
        cmd = '"{:s}"'.format(LINUX_CMDS.loc[i]['command'])
        mutator_list = [MUTATORS[random.randint(0, len(MUTATORS) - 1)]]
        bashfuscated = bashfuscate(cmd, mutator_list)
        # dataset.append([1, mutator_list[0], bashfuscated.stdout])
        dataset.append([1, bashfuscated.stdout])
    except Exception as e:
        print('error')
        print(e)
        unparseable += 1
print('done 1 type')

random_2_layer_idx = random.sample(range(NUM_CMDS), 100)
print(random_2_layer_idx[0:5])
x = 0
for i in random_2_layer_idx:
    if x % 10 == 0:
        print(x)
    x += 1

    try:
        cmd = '"{:s}"'.format(LINUX_CMDS.loc[i]['command'])
        mutator_list = [MUTATORS[random.randint(0, len(MUTATORS) - 1)] for _ in range(2)]
        bashfuscated = bashfuscate(cmd, mutator_list)
        # dataset.append([1, ', '.join(mutator_list), bashfuscated.stdout])
        dataset.append([1, bashfuscated.stdout])
    except Exception as e:
        print('error')
        print(e)
        unparseable += 1
print('done 2 type')

# df = pd.DataFrame(dataset, columns=['label', 'obf technique', 'command'])
df = pd.DataFrame(dataset, columns=['label', 'command'])
df.to_csv(DATA_DIR + CSV_DIR + 'linux-data.csv', index=False)

# sanity checks
print(len(df))
print('unparseable:', unparseable)
