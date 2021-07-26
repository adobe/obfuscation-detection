import torch

DATA_DIR = '../data/prep/'
FREQ_CUTOFF = 0.00002 # found from char_frequency.py to include ^

char_freq_file = open(DATA_DIR + 'char_freq.txt', 'r')
char_dict = {}
char_dict_curr_idx = 0

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

torch.save(char_dict, DATA_DIR + 'char_dict.pth')
