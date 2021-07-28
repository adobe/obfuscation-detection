import torch
import re

# from https://github.com/first20hours/google-10000-english, removed a-z letters
top_10k_file = open('../data/top-10k-words.txt')
top_10k_words = set(top_10k_file.read().split())
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

print(len(top_10k_words))

commands = torch.load('../data/scripts/all_scripts.pth')
print(len(commands))
# print(commands[0])
# command = ''.join(e for e in commands[0] if e.isalnum() or e == ' ')
# print(command)
# tokens = command.split()
# print(tokens)
# known = 0
# for token in tokens:
#     token = token.lower()
#     if token in top_10k_words:
#         known += 1
# print(known)
# print(len(tokens))

ratios = []
for command in commands:
    command = re.sub(r'\W+', ' ', command)
    tokens = command.split()
    if len(tokens) == 0:
        ratios.append(0)
        continue
    known = 0
    for token in tokens:
        token = token.lower()
        if token in top_10k_words:
            known += 1
    ratios.append(known / len(tokens))

print(len(ratios))
print(ratios[0])

sorted_idx = sorted(range(len(ratios)), key = lambda k: ratios[k], reverse = True)
ranked_file = open('../data/unk_word_ratio_ranked_only_cmd.txt', 'w')
for i in sorted_idx:
    ranked_file.write('ratio: ' + str(ratios[i]) + '\n')
    ranked_file.write(commands[i] + '\n')


# print(ratios[len(ratios) - 1])
# command = commands[sorted_idx[len(ratios) - 1]]
# print(command)
# command = re.sub(r'\W+', ' ', command)
# print(command)
# tokens = command.split()
# print(tokens)
# known = 0
# for token in tokens:
#     token = token.lower()
#     if token in top_10k_words:
#         known += 1
# print(known)
# print(len(tokens))
# print(str(known / len(tokens)))

for i in sorted_idx[0:5]:
    print(ratios[i])
    print(re.sub(r'\W+', ' ', commands[i]).split())
    print()
for i in sorted_idx[len(ratios)-5:]:
    print(ratios[i])
    print(re.sub(r'\W+', ' ', commands[i]).split())
    print()