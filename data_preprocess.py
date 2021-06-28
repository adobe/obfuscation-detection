import pandas as pd
import torch
import traceback

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
    'UnderhandedPowerShell-obfuscation-labeledData.csv'
]
PROCESSED_TENSORS_DIR = 'data/processed_tensors/'
FREQ_CUTOFF = 0.0003 # found from char_frequency.py
TENSOR_LENGTH = 1024
# cross reference powershell https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_character_encoding?view=powershell-7.1 
# and python https://docs.python.org/2.7/library/codecs.html#standard-encodings
ENCODING_TYPES = ['ascii', 'utf_16_be', 'utf_16_le', 'utf_16', 'utf_7', 'utf_8', 'utf_32', 'utf_32_be', 'utf_32_le', 'utf_8_sig']

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
        char_dict[char] = char_dict_curr_idx
        char_dict_curr_idx += 1
print(char_dict)
print('len of chars:', len(char_dict))

# iterate through labels
x = 0
unparseable = 0
for label_file in LABEL_FILES:
    csv = pd.read_csv(LABELS_DIR + label_file)
    # iterate through files
    for j, row in csv.iterrows():
        if x % 1000 == 0:
            print(x)
        x += 1

        ps_path = row[0].replace('\\', '/') # windows to mac file reading
        # parse powershell script
        parsed = False
        for codec in ENCODING_TYPES:
            try:
                ps_file = open(DATA_DIR + ps_path, encoding=codec)
                ps_contents = ''.join(ps_file.readline())
                parsed = True
            except:
                pass
        # try:
        #     ps_file = open(DATA_DIR + ps_path, encoding='utf-8')
        #     ps_contents = ''.join(ps_file.readlines())
        # except:
        #     ps_file = open(DATA_DIR + ps_path, encoding='utf-16')
        #     ps_contents = ''.join(ps_file.readlines())
        if parsed:
            # construct one tensor for one whole powershell script
            try:
                ps_tensor = torch.zeros(len(char_dict) + 1, TENSOR_LENGTH) # + 1 for case bit
                tensor_len = min(len(ps_contents), TENSOR_LENGTH)
                for i in range(tensor_len):
                    if ps_contents[i] in char_dict or (ps_contents[i].isupper() and ps_contents[i].lower() in char_dict):
                        if ps_contents[i].isupper():
                            ps_tensor[len(char_dict)][i] = 1 # set upper case bit
                        ps_tensor[char_dict[ps_contents[i].lower()]][i] = 1
                # add to overall list
                converted_tensors.append(ps_tensor)
                label = int(row[1])
                if label == 1:
                    tensor_labels.append(torch.Tensor([0, 1]))
                else:
                    tensor_labels.append(torch.Tensor([1, 0]))
            except Exception as e:
                print(ps_contents[i])
                traceback.print_exc()
                print(e)
        else:
            unparseable += 1
print('unparseable files: {:d}'.format(unparseable))

# print(ps_tensor[0])
# print(ps_tensor[0].nonzero())
# print(ps_tensor[1].nonzero())
# print(ps_tensor[2].nonzero())
# print(ps_tensor[3].nonzero())
# print(ps_tensor.shape)

# sanity checks
print(converted_tensors[0].shape)
print(converted_tensors[0][23][0])
print(converted_tensors[0][69][0])
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
                                                            [8430, 1580, 526], # hard-calculated 80-15-5 split
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
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
train_y = torch.stack(train_y)
val_x = torch.stack(val_x)
val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1], val_x.shape[2])
val_y = torch.stack(val_y)
test_x = torch.stack(test_x)
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2])
test_y = torch.stack(test_y)

print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape)

# save tensors into files
# torch.save(train_x, PROCESSED_TENSORS_DIR + 'train_x.pt')
# torch.save(train_y, PROCESSED_TENSORS_DIR + 'train_y.pt')
# torch.save(val_x, PROCESSED_TENSORS_DIR + 'val_x.pt')
# torch.save(val_y, PROCESSED_TENSORS_DIR + 'val_y.pt')
# torch.save(test_x, PROCESSED_TENSORS_DIR + 'test_x.pt')
# torch.save(test_y, PROCESSED_TENSORS_DIR + 'test_y.pt')

torch.save({'x': train_x, 'y': train_y}, PROCESSED_TENSORS_DIR + 'train_data.pt')
torch.save({'x': val_x, 'y': val_y}, PROCESSED_TENSORS_DIR + 'val_data.pt')
torch.save({'x': test_x, 'y': test_y}, PROCESSED_TENSORS_DIR + 'test_data.pt')
