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
obf_csv = pd.read_csv(DATA_DIR + 'new-positives.csv', sep='\t')
print(obf_csv.shape[0])

dataset = []
x = 0
for i in range(obf_csv.shape[0]):
    if x % 10000 == 0:
        print(x)
    x += 1

    dataset.append([1, obf_csv.loc[i]['process']])

df = pd.DataFrame(dataset, columns=['label', 'command'])
df.to_csv(DATA_DIR + CSV_DIR + 'obf-data.csv', index=False)

# sanity checks
print(len(df))
