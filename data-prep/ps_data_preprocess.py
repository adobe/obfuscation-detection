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
import re

# dataset from Bohannon (2017)
DATA_DIR = '../data/'
CSV_DIR = 'processed_csv/'
PS_DIR = DATA_DIR + 'PowerShellCorpus/'
LABELS_DIR = '../../Revoke-Obfuscation/DataScience/'
LABEL_FILES = [
    'GithubGist-obfuscation-labeledData.csv',
    'InvokeCradleCrafter-obfuscation-labeledData.csv',
    'InvokeObfuscation-obfuscation-labeledData.csv',
    'IseSteroids-obfuscation-labeledData.csv',
    'PoshCode-obfuscation-labeledData.csv',
    'TechNet-obfuscation-labeledData.csv',
    # 'UnderhandedPowerShell-obfuscation-labeledData.csv'
]
PROCESSED_TENSORS_DIR = DATA_DIR + 'processed_tensors/'
SCRIPTS_DIR = DATA_DIR + 'scripts/'

# iterate through labels
dataset = []
x = 0
unparseable, num_pos = 0, 0
filenames = []
for label_file in LABEL_FILES:
    csv = pd.read_csv(LABELS_DIR + label_file)
    # iterate through files
    for _, row in csv.iterrows():
        if x % 1000 == 0:
            print(x)
        x += 1

        ps_path = row[0].replace('\\', '/') # windows to mac file reading
        '''
        # only for filtering it out
        # skip "real-world" obfuscated b/c it's not that obfuscated in the dataset
        if (ps_path.startswith('GithubGist') or ps_path.startswith('PoshCode') or ps_path.startswith('TechNet')) and \
            int(row[1]) == 1:
            continue
        '''

        # parse powershell script

        # filter out non-utf8 characters
        try:
            ps_file = open(PS_DIR + ps_path, 'rb')
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
        is_obf = 1 if int(row[1]) == 1 else 0
        if is_obf == 1:
            num_pos += 1
        
        dataset.append([is_obf, file_str])
        filenames.append(ps_path)

df = pd.DataFrame(dataset, columns=['label', 'command'])
df.to_csv(DATA_DIR + CSV_DIR + 'ps-data.csv', index=False)

# sanity checks
print('unparseable files: {:d}'.format(unparseable))
print('num pos:', num_pos)
print(len(df))
