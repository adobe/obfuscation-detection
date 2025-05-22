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

DATA_DIR = '../data/'
CSV_DIR = 'processed_csv/'
dos_files = [open(DATA_DIR + 'STATIC_1-of-4_Out-DosConcatenatedCommand.txt', 'r', encoding='utf-16'),
            open(DATA_DIR + 'STATIC_2-of-4_Out-DosReversedCommand.txt', 'r', encoding='utf-16'),
            open(DATA_DIR + 'STATIC_3-of-4_Out-DosFORcodedCommand.txt', 'r', encoding='utf-16'),
            open(DATA_DIR + 'STATIC_4-of-4_Out-DosFINcodedCommand.txt', 'r', encoding='utf-16')]
dos_lines = []
for dos_file in dos_files:
    dos_lines += dos_file.read().splitlines()
print(dos_lines[0])
print(len(dos_lines[0]))
print(len(dos_lines))

dataset = []

for line in dos_lines:
    dataset.append([1, line])

df = pd.DataFrame(dataset, columns=['label', 'command'])
df.to_csv(DATA_DIR + CSV_DIR + 'dos-data.csv', index=False)

# sanity checks
print(len(df))
