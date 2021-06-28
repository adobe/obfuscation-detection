import pandas as pd

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

char_freq_file = open('char_freq.txt', 'x')
char_counts = {}
total_chars = 0
unparsable = 0
total_cmds = 0

# iterate through labels
for label_file in LABEL_FILES:
    csv = pd.read_csv(LABELS_DIR + label_file)
    # iterate through files
    for _, row in csv.iterrows():
        total_cmds += 1
        ps_path = row[0].replace('\\', '/') # windows to mac file reading
        if total_cmds % 1000 == 0:
            print(total_cmds)
        try:
            try:
                ps_file = open(DATA_DIR + ps_path, encoding='utf-8')
                ps_contents = ''.join(ps_file.readlines())
            except:
                ps_file = open(DATA_DIR + ps_path, encoding='utf-16')
                ps_contents = ''.join(ps_file.readlines())
            for char in ps_contents:
                if char.isalpha():
                    char = char.lower()
                if char not in char_counts:
                    char_counts[char] = 0
                char_counts[char] += 1
                total_chars += 1
        except:
            unparsable += 1
for char in char_counts:
    char_counts[char] /= total_chars
for k, v in sorted(char_counts.items(), key=lambda p:p[1], reverse=True):
    char_freq_file.write(k + ' ' + str(v) + '\n')

print('total commands:', total_cmds)
print('unparseable commands:', unparsable)
print('parseable commands:', total_cmds - unparsable)

# current cutoff is 0.0003691787536410754, or 0.0003
