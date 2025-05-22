import pandas as pd

DATA_DIR = '../data/'
CSV_DIR = '../data/processed_csv/'

win_csv = pd.read_csv(DATA_DIR + 'val-win-data-1500.csv').values.tolist()
linux_csv = pd.read_csv(DATA_DIR + 'val-linux-data-31k.csv').values.tolist()

dataset = []

for row in win_csv:
    dataset.append([0, row[0]])

for row in linux_csv[0:1500]:
    dataset.append([0, '"' + row[0] + '"'])

df = pd.DataFrame(dataset, columns=['label', 'command'])
df.to_csv(CSV_DIR + 'extra-val-data.csv', index=False)

print(len(df))