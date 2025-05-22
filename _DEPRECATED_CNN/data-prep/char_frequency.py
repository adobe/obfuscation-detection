# 
# Authors: Security Intelligence Team within the Security Coordination Center
# 
# Copyright 2021 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS 
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#

import pandas as pd
import traceback

DATA_DIR = '../data/PowerShellCorpus/'
LABELS_DIR = '../../Revoke-Obfuscation/DataScience/'
PREP_DIR = '../data/prep/'
LABEL_FILES = [
    'GithubGist-obfuscation-labeledData.csv',
    'InvokeCradleCrafter-obfuscation-labeledData.csv',
    'InvokeObfuscation-obfuscation-labeledData.csv',
    'IseSteroids-obfuscation-labeledData.csv',
    'PoshCode-obfuscation-labeledData.csv',
    'TechNet-obfuscation-labeledData.csv',
    # 'UnderhandedPowerShell-obfuscation-labeledData.csv', # not provided to us by author
]

char_freq_file = open(PREP_DIR + 'char_freq.txt', 'x')
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
