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

DATA_DIR = '../data/processed_csv/'

train_rows = pd.read_csv(DATA_DIR + 'win-train-data.csv').values.tolist()
linux_train_rows = pd.read_csv(DATA_DIR + 'linux-train-data.csv').values.tolist()
train_rows += [[row[0], row[2]] for row in linux_train_rows]
print('loaded train rows', len(train_rows))

val_rows = pd.read_csv(DATA_DIR + 'win-val-data.csv').values.tolist()
linux_val_rows = pd.read_csv(DATA_DIR + 'linux-val-data.csv').values.tolist()
val_rows += [[row[0], row[2]] for row in linux_val_rows]
print('loaded val rows', len(val_rows))

test_rows = pd.read_csv(DATA_DIR + 'win-test-data.csv').values.tolist()
linux_test_rows = pd.read_csv(DATA_DIR + 'linux-test-data.csv').values.tolist()
test_rows += [[row[0], row[2]] for row in linux_test_rows]
print('loaded test rows', len(test_rows))

train_df = pd.DataFrame(train_rows, columns=['label', 'command'])
val_df = pd.DataFrame(val_rows, columns=['label', 'command'])
test_df = pd.DataFrame(test_rows, columns=['label', 'command'])

train_df.to_csv(DATA_DIR + 'all-train-data.csv', index=False)
val_df.to_csv(DATA_DIR + 'all-val-data.csv', index=False)
test_df.to_csv(DATA_DIR + 'all-test-data.csv', index=False)

# sanity checks
print(len(train_df), len(val_df), len(test_df))
print(len(train_df) + len(val_df) + len(test_df))
