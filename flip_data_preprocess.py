import pandas as pd
import torch
import traceback
import os
import re

# dataset from Bohannon (2017)
DATA_DIR = 'data/PowerShellCorpus/'
LABELS_DIR = '../Revoke-Obfuscation/DataScience/'
LABEL_FILES = [
    'GithubGist-obfuscation-labeledData.csv',
    'InvokeCradleCrafter-obfuscation-labeledData.csv',
    'InvokeObfuscation-obfuscation-labeledData.csv',
    'IseSteroids-obfuscation-labeledData.csv',
    'PoshCode-obfuscation-labeledData.csv',
    'TechNet-obfuscation-labeledData.csv',
    # 'UnderhandedPowerShell-obfuscation-labeledData.csv'
]
PROCESSED_TENSORS_DIR = 'data/processed_tensors/'
FREQ_CUTOFF = 0.0003 # found from char_frequency.py
TENSOR_LENGTH = 4096

char_freq_file = open('char_freq.txt', 'r')
char_dict = {}
char_dict_curr_idx = 0
converted_tensors = []
tensor_labels = []

# make character to tensor index dictionary from the char frequencies file
for char_freq in char_freq_file:
    char_freq = char_freq.split(' ')
    char = char_freq[0]
    freq = float(char_freq[1][:-1]) # remove new line
    if freq > FREQ_CUTOFF:
        if char == '<space>':
            char = ' '
        elif char == '<tab>':
            char = '\t'
        elif char == '<newline>':
            char = '\n'
        elif char == '<return>':
            char = '\r'
        char_dict[char] = char_dict_curr_idx
        char_dict_curr_idx += 1
print(char_dict)
print('len of chars:', len(char_dict))

# iterate through labels
x = 0
unparseable, num_pos, num_neg = 0, 0, 0
filenames = []
for label_file in LABEL_FILES:
    csv = pd.read_csv(LABELS_DIR + label_file)
    # iterate through files
    for _, row in csv.iterrows():
        if x % 1000 == 0:
            print(x)
        x += 1

        ps_path = row[0].replace('\\', '/') # windows to mac file reading
        # skip "real-world" obfuscated b/c it's not that obfuscated in the dataset
        if (ps_path.startswith('GithubGist') or ps_path.startswith('PoshCode') or ps_path.startswith('TechNet')) and \
            int(row[1]) == 1:
            continue

        # parse powershell script

        # filter out non-utf8 characters
        try:
            ps_file = open(DATA_DIR + ps_path, 'rb')
            file_str = ''
            byte = ps_file.read(1)
            while byte:
                if byte == b'\x00':
                    # skip this null byte
                    byte = ps_file.read(1)
                    continue
                try:
                    byte_str = str(byte, 'utf-8')
                except:
                    # non-utf8 byte
                    byte = ps_file.read(1)
                    continue
                # valid utf-8 byte
                file_str += byte_str
                byte = ps_file.read(1)
            ps_file.close()
            filenames.append(ps_path)
        except Exception as e: 
            unparseable += 1
            traceback.print_exc()
            print(e)
            continue

        file_str_split = file_str.splitlines()

        # filter out multi-line comments
        # only looks at multi-line comment start/end points if they are on their own line
        # only comments like '<#\nasdfasdf\nasdfasdf\n#>' or '  <# \nasdfasdf\nasdfasdf\n  #> ' 
        # NOTE: do this before filtering out single line comments b/c '#>' looks like a single-line comment
        multi_line_indices = []
        start = -1
        for i in range(len(file_str_split)):
            line = file_str_split[i]
            if re.match('[ \t]*<#[ \t]*', line):
                start = i
            elif re.match('[ \t]*#>[ \t]*', line) and start != -1:
                multi_line_indices += range(start, i + 1)
                start = -1
        for i in multi_line_indices[::-1]:
            del file_str_split[i]

        # filter out single line comments
        # only lines like '# asdf' or '  # asdf', not inline comments
        file_str_split = [i for i in file_str_split if not re.match('[ \t]*#.*', i)]

        # convert file string into tensor
        file_str = '\n'.join(file_str_split)
        # if ps_path == 'GithubGist/9to5IT_9620565_raw_04b5a0e0d62290ccf025de4ab9c75597a75d6d9c_Logging_Functions.ps1':
        #     print(multi_line_indices)
        #     print(file_str)
        ps_tensor = torch.zeros(TENSOR_LENGTH, len(char_dict) + 1) # + 1 for case bit
        tensor_len = min(len(file_str), TENSOR_LENGTH)
        for i in range(tensor_len):
            char = file_str[i]
            lower_char = char.lower()
            if char.isupper() and lower_char in char_dict:
                ps_tensor[i][len(char_dict)] = 1
                char = lower_char
            if char in char_dict:
                ps_tensor[i][char_dict[char]] = 1
        converted_tensors.append(ps_tensor)
        if int(row[1]) == 1:
            num_pos += 1
            tensor_labels.append(torch.Tensor([0, 1]))
        else:
            num_neg += 1
            tensor_labels.append(torch.Tensor([1, 0]))


# print(ps_tensor[0])
# print(ps_tensor[0].nonzero())
# print(ps_tensor[1].nonzero())
# print(ps_tensor[2].nonzero())
# print(ps_tensor[3].nonzero())
# print(ps_tensor.shape)

# sanity checks
print('unparseable files: {:d}'.format(unparseable))
print(num_pos, num_neg)
print(converted_tensors[0].shape)
print(converted_tensors[0][0][23])
print(converted_tensors[0][0][70])
print(converted_tensors[0][1][14])
print(converted_tensors[0][1][70])

print(tensor_labels[0])
print(len(converted_tensors))
print(len(tensor_labels))

# for i in random.sample(range(len(filenames)), 50):
#     print(filenames[i], tensor_labels[i])
train_filenames = []
val_filenames = []
test_filenames = []

# split all data into train, val, test
train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []
train_idx, val_idx, test_idx = torch.utils.data.random_split(range(len(converted_tensors)), 
                                                            # hard-calculated 80-15-5 split
                                                            # [8704, 1632, 544], # all samples - 10880
                                                            # [7745, 1452, 484], # no real-world - 9681
                                                            [7704, 1444, 481], # no real-world + data cleaned - 9629
                                                            generator=torch.Generator().manual_seed(42))
for i in train_idx:
    train_x.append(converted_tensors[i])
    train_y.append(tensor_labels[i])
    train_filenames.append(filenames[i])
for i in val_idx:
    val_x.append(converted_tensors[i])
    val_y.append(tensor_labels[i])
    val_filenames.append(filenames[i])
for i in test_idx:
    test_x.append(converted_tensors[i])
    test_y.append(tensor_labels[i])
    test_filenames.append(filenames[i])
train_x = torch.stack(train_x)
train_y = torch.stack(train_y)
val_x = torch.stack(val_x)
val_y = torch.stack(val_y)
test_x = torch.stack(test_x)
test_y = torch.stack(test_y)

print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape)

torch.save({'x': train_x, 'y': train_y}, PROCESSED_TENSORS_DIR + 'flip_train_data.pth')
torch.save({'x': val_x, 'y': val_y}, PROCESSED_TENSORS_DIR + 'flip_val_data.pth')
torch.save({'x': test_x, 'y': test_y}, PROCESSED_TENSORS_DIR + 'flip_test_data.pth')
torch.save(char_dict, 'char_dict.pth')
torch.save(train_filenames, 'train_filenames_list.pth')
torch.save(val_filenames, 'val_filenames_list.pth')
torch.save(test_filenames, 'test_filenames_list.pth')

# for i in random.sample(range(len(val_filenames)), 20):
#     print(val_filenames[i], int(torch.argmax(val_y[i])))
