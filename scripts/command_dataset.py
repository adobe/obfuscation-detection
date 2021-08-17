import torch
import pandas as pd

DATA_DIR = '../data/'
TENSOR_LENGTH = 4096

class CommandDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.char_dict = torch.load(DATA_DIR + 'prep/char_dict.pth')
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        cmd = str(row['command'])
        label = int(row['label'])

        x = torch.zeros(len(self.char_dict) + 1, TENSOR_LENGTH, dtype=torch.int8)
        tensor_len = min(TENSOR_LENGTH, len(cmd))
        # label x
        for j in range(tensor_len):
            char = cmd[j]
            lower_char = char.lower()
            if char.isupper() and lower_char in self.char_dict:
                x[len(self.char_dict)][j] = 1
                char = lower_char
            if char in self.char_dict:
                x[self.char_dict[char]][j] = 1
        
        # label y
        if label == 1:
            y = torch.tensor([0, 1], dtype=torch.int8)
        else :
            y = torch.tensor([1, 0], dtype=torch.int8)

        return x, y
