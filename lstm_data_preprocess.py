import pandas as pd
import torch
import traceback
import os

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
FREQ_CUTOFF = 0.0002 # found from char_frequency.py
TENSOR_LENGTH = 1024

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
for label_file in LABEL_FILES:
    csv = pd.read_csv(LABELS_DIR + label_file)
    # iterate through files
    for _, row in csv.iterrows():
        if x % 1000 == 0:
            print(x)
        x += 1

        ps_path = row[0].replace('\\', '/') # windows to mac file reading
        # parse powershell script
        try:
            ps_file = open(DATA_DIR + ps_path, 'rb')
            ps_tensor = torch.zeros(TENSOR_LENGTH, len(char_dict) + 1) # + 1 for case bit
            tensor_len = min(os.path.getsize(DATA_DIR + ps_path), TENSOR_LENGTH)

            for i in range(tensor_len):
                byte = ps_file.read(1)
                try:
                    byte_char = byte.decode('utf-8')
                except:
                    continue # invalid char
                # check for uppercase
                lower_char = byte_char.lower()
                if byte_char.isupper() and lower_char in char_dict:
                    ps_tensor[i][len(char_dict)] = 1
                    byte_char = lower_char
                # check if byte is most frequent
                if byte_char in char_dict:
                    ps_tensor[i][char_dict[byte_char]] = 1
            # add to data points to list
            converted_tensors.append(ps_tensor)
            if int(row[1]) == 1:
                num_pos += 1
                tensor_labels.append(torch.Tensor([0, 1]))
            else:
                num_neg += 1
                tensor_labels.append(torch.Tensor([1, 0]))
            ps_file.close()
        except Exception as e:
            traceback.print_exc()
            print(e)

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
print(converted_tensors[0][0][24])
print(converted_tensors[0][0][71])
print(tensor_labels[0])
print(len(converted_tensors))
print(len(tensor_labels))

# split all data into train, val, test
train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []
train_idx, val_idx, test_idx = torch.utils.data.random_split(range(len(converted_tensors)), 
                                                            [8704, 1632, 544], # hard-calculated 80-15-5 split
                                                            generator=torch.Generator().manual_seed(42))
for i in train_idx:
    train_x.append(converted_tensors[i])
    train_y.append(tensor_labels[i])
for i in val_idx:
    val_x.append(converted_tensors[i])
    val_y.append(tensor_labels[i])
for i in test_idx:
    test_x.append(converted_tensors[i])
    test_y.append(tensor_labels[i])
train_x = torch.stack(train_x)
train_y = torch.stack(train_y)
val_x = torch.stack(val_x)
val_y = torch.stack(val_y)
test_x = torch.stack(test_x)
test_y = torch.stack(test_y)

print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape)

torch.save({'x': train_x, 'y': train_y}, PROCESSED_TENSORS_DIR + 'lstm_train_data.pth')
torch.save({'x': val_x, 'y': val_y}, PROCESSED_TENSORS_DIR + 'lstm_val_data.pth')
torch.save({'x': test_x, 'y': test_y}, PROCESSED_TENSORS_DIR + 'lstm_test_data.pth')
torch.save(char_dict, 'char_dict.pth')
