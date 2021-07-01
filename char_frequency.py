import pandas as pd
import traceback

DATA_DIR = 'data/PowerShellCorpus/'
LABELS_DIR = '../Revoke-Obfuscation/DataScience/'
LABEL_FILES = [
    'GithubGist-obfuscation-labeledData.csv',
    'InvokeCradleCrafter-obfuscation-labeledData.csv',
    'InvokeObfuscation-obfuscation-labeledData.csv',
    'IseSteroids-obfuscation-labeledData.csv',
    'PoshCode-obfuscation-labeledData.csv',
    'TechNet-obfuscation-labeledData.csv',
    # 'UnderhandedPowerShell-obfuscation-labeledData.csv', # not provided to us by author
]

char_freq_file = open('char_freq.txt', 'x')
char_counts = {}
total_chars = 0
unparsable = 0
total_cmds = 0

# iterate through labels
for label_file in LABEL_FILES:
    csv = pd.read_csv(LABELS_DIR + label_file, encoding='utf-8')
    # iterate through files
    for _, row in csv.iterrows():
        total_cmds += 1
        if total_cmds % 1000 == 0:
            print(total_cmds)
        ps_path = row[0].replace('\\', '/') # windows to mac file reading
        try:
            # iterate through each byte and increment the count
            ps_file = open(DATA_DIR + ps_path, 'rb')
            byte = ps_file.read(1)
            while byte:
                if byte == b'\x00':
                    # skip this null byte
                    byte = ps_file.read(1)
                    continue
                try:
                    byte_str = str(byte, 'utf-8')
                    # if upper, add to count for lower char
                    if byte_str.isalpha() and byte_str.isupper():
                        byte = byte_str.lower().encode('utf-8')
                except:
                    byte = ps_file.read(1)
                    continue
                if byte not in char_counts:
                    char_counts[byte] = 0
                char_counts[byte] += 1
                total_chars += 1
                byte = ps_file.read(1)
            ps_file.close()
        except Exception as e:
            traceback.print_exc()
            print(e)
            unparsable += 1
for char in char_counts:
    char_counts[char] /= total_chars
for k, v in sorted(char_counts.items(), key=lambda p:p[1], reverse=True):
    try:
        char_freq_file.write(k.decode('utf-8') + ' ' + str(v) + '\n')
    except:
        # skip non utf-8 characters
        pass

print('total commands:', total_cmds)
print('unparseable commands:', unparsable)
print('parseable commands:', total_cmds - unparsable)

char_freq_file.close()
# current cutoff is 0.0003691787536410754, or 0.0003
