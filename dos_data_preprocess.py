import torch

TENSOR_LENGTH = 4096
CHAR_DICT = torch.load('char_dict.pth')
dos_file = open('data/STATIC_1-of-4_Out-DosConcatenatedCommand.txt', 'r', encoding='utf-16')

print(CHAR_DICT)
dos_lines = dos_file.readlines()
print(dos_lines[0])
print(len(dos_lines[0]))
print(len(dos_lines))

tensors_x = []
tensors_y = [torch.tensor([1, 0]) for _ in range(len(dos_lines))]
scripts = []

for line in dos_lines:
    script_tensor = torch.zeros(len(CHAR_DICT) + 1, TENSOR_LENGTH)
    tensor_len = min(TENSOR_LENGTH, len(line))

    for i in range(tensor_len):
        char = line[i]
        lower_char = char.lower()
        if char.isupper() and lower_char in CHAR_DICT:
            script_tensor[len(CHAR_DICT)][i] = 1
            char = lower_char
        if char in CHAR_DICT:
            script_tensor[CHAR_DICT[char]][i] = 1
    
    tensors_x.append(script_tensor)
    scripts.append(line)

tensors_x = torch.stack(tensors_x)
tensors_y = torch.stack(tensors_y)

print(tensors_x[0][23][0])
print(tensors_x[0][73][0])
print(tensors_x[0][23][5])
print(tensors_x[0][73][5])
print(tensors_y[0])
print(tensors_x.shape)
print(tensors_y.shape)

torch.save({'x': tensors_x, 'y': tensors_y}, 'data/processed_tensors/dos_data.pth')
torch.save(scripts, 'dos_scripts.pth')