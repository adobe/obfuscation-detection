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

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from models import *
from command_dataset import CommandDataset

DATA_DIR = '../data/processed_csv/'
SCRIPTS_DIR = '../data/scripts/'
PREP_DIR = '../data/prep/'
EPOCHS = 100
BATCH_SIZE = 64

model = ShallowCNN()
model_file = 'models/cnn-shallow-conv-1-fc-2048-1024.pth'
cuda_device = 0

# set random seed for reproducibility
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='obfuscation detection train file')
parser.add_argument('--reset', help='start over training', action='store_true')
parser.add_argument('--eval', help='eval model', action='store_true')
parser.add_argument('--analyze', help='analyze statistics and samples of model', action='store_true')
parser.add_argument('--run', help='run model on a real script', action='store_true')
parser.add_argument('--test', help='eval model on test set', action='store_true')
parser.add_argument('--model', default='cnn', help='model to run (mlp, deep-mlp, cnn, deep-cnn)')
parser.add_argument('--model_file', default='cnn-shallow-conv-1-fc-2048-1024.pth', help='model file to save/load')
parser.add_argument('--cuda_device', default=0, type=int, help='which cuda device to use')
args = parser.parse_args()

if args.model == 'mlp':
    print('using MLP model')
    model = MLP()
elif args.model == 'deep-mlp':
    print('using deep MLP model')
    model = DeepMLP()
elif args.model == 'cnn':
    print('using CNN model')
    model = ShallowCNN()
elif args.model == 'deep-cnn':
    print('using deep CNN model')
    model = DeepCNN()
elif args.model == 'cnn-2':
    print('using CNN 2')
    model = Conv2()
elif args.model == 'cnn-3':
    print('using CNN 3')
    model = Conv3()
elif args.model == 'cnn-4':
    print('using CNN 4')
    model = Conv4()
elif args.model == 'cnn-5':
    print('using CNN 5')
    model = Conv5()
elif args.model == 'large-cnn-2':
    print('using Large CNN 2')
    model = LargeCNN2()
elif args.model == 'cnn-gated':
    print('using Gated CNN')
    model = GatedCNN()
elif args.model == 'cnn-2-gated':
    print('using Gated CNN 2')
    model = Conv2Gated()
elif args.model == 'lstm-simple':
    print('using simple LSTM')
    model = SimpleLSTM()
elif args.model == 'lstm-small':
    print('using small LSTM')
    model = SmallLSTM()
elif args.model == 'lstm-large':
    print('using large LSTM')
    model = LargeLSTM()
elif args.model == 'resnet':
    print('using resnet')
    model = ResNet()

model_file = '../models/' + args.model_file

device = torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)
    device = torch.device('cuda')
    print('using CUDA', torch.cuda.current_device())

if args.eval and args.reset:
    print('cannot eval without loading model')
    exit(1)

# init model
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
mse = nn.MSELoss()
epoch = 0

# load data
train_data = CommandDataset(DATA_DIR + 'train-data.csv')
val_data = CommandDataset(DATA_DIR + 'val-data.csv')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
print('loaded data:', len(train_data), len(val_data))

best_val_f1 = 0.
if not args.reset:
    # load checkpoint if eval or not retraining
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    if 'val_f1' in checkpoint:
        val_f1 = checkpoint['val_f1']
        best_val_f1 = val_f1
        print('loaded {:s} on epoch {:d} with val f1 {:f}'.format(model_file, epoch, val_f1))
    else:
        print('loaded model with acc ' + str(checkpoint['val_acc']))

def eval_model(dataset_name, model, data_loader, num_data, loss_fn):
    model.eval()
    avg_loss = 0.
    num_batches = 0
    y_true = []
    y_pred = []
    for _, (data, label) in enumerate(data_loader):
        # run model
        data, label = Variable(data.type(torch.float).to(device)), \
                            Variable(label.type(torch.float).to(device))
        output = model(data)

        # calculate loss
        loss = loss_fn(output, label)
        avg_loss += loss.data
        num_batches += 1
        # calculate accuracy
        y_pred += list(torch.max(output, dim=1).indices.cpu().numpy())
        y_true += list(torch.max(label, dim=1).indices.cpu().numpy())
    avg_loss /= num_batches
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('\t{:s} loss: {:f}'.format(dataset_name, avg_loss))
    print('\t{:s} accuracy: {:f}'.format(dataset_name, acc))
    print('\t{:s} f1: {:f}'.format(dataset_name, f1))
    print('\t{:s} precision: {:f}'.format(dataset_name, prec))
    print('\t{:s} recall: {:f}'.format(dataset_name, recall))
    print('\t{:s} confusion matrix (tn, fp, fn, tp): {:d}, {:d}, {:d}, {:d}'.format(dataset_name, tn, fp, fn, tp))
    return avg_loss, f1

if args.eval:
    # eval model
    eval_model('train', model, train_loader, len(train_data), mse)
    eval_model('val', model, val_loader, len(val_data), mse)
elif args.test:
    # eval model on test
    test_data = CommandDataset(torch.load(DATA_DIR + 'test_data.pth'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    eval_model('test', model, test_loader, len(test_data), mse)
elif args.analyze:        
    model.eval()
    CSV_FILE = 'val-data.csv'
    eval_csv = pd.read_csv(DATA_DIR + CSV_FILE)
    # eval_data = CommandDataset(DATA_DIR + CSV_FILE)
    # eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=False)
    fn_file = open('fn.txt', 'w')
    fp_file = open('fp.txt', 'w')
    tp_file = open('tp.txt', 'w')
    tp_file.write('process\n')

    # print(len(eval_loader))
    # print(len(eval_data))

    # analyze on all val samples
    y_true = []
    y_pred = []
    for i, (data, label) in enumerate(val_loader):
        if i % 1000 == 0:
            print(i)
        
        data, label = Variable(data.type(torch.float).to(device)), \
                            Variable(label.type(torch.float).to(device))
        output = model(data)

        curr_y_pred = list(torch.max(output, dim=1).indices.cpu().numpy())
        curr_y_true = list(torch.max(label, dim=1).indices.cpu().numpy())
        for j in range(len(curr_y_pred)):
            curr_idx = i * BATCH_SIZE + j
            # print(i, j)
            # false negatives
            if curr_y_pred[j] == 0 and curr_y_true[j] == 1:
                fn_file.write('\nScript {:d}\n'.format(curr_idx))
                fn_file.write(eval_csv.loc[curr_idx]['command'])
            
                # fn_file.write('"{:s}"\n'.format(commands[curr_idx]))
            
            # false positives
            if curr_y_pred[j] == 1 and curr_y_true[j] == 0:                
                fp_file.write('\nScript {:d}\n'.format(curr_idx))
                fp_file.write(eval_csv.loc[curr_idx]['command'])
                
                # fp_file.write('"{:s}"\n'.format(commands[curr_idx]))

            # true positives
            if curr_y_pred[j] == 1 and curr_y_true[j] == 1:                
                tp_file.write('\nScript {:d}\n'.format(curr_idx))
                tp_file.write(eval_csv.loc[curr_idx]['command'])

                # tp_file.write('"{:s}"\n'.format(commands[curr_idx]))

        y_pred += curr_y_pred
        y_true += curr_y_true

    '''
    # analyze on random 100 samples
    data = []
    label = []
    sampled_filenames = []
    for i in range(0, 300):
        x, y = val_data[i]
        data.append(x)
        label.append(y)
        sampled_filenames.append(val_filenames[i])
    data = Variable(torch.stack(data))
    label = Variable(torch.stack(label))
    output = model(data)
    y_pred = list(torch.max(output, dim=1).indices.cpu().numpy())
    y_true = list(torch.max(label, dim=1).indices.cpu().numpy())

    for i in range(len(y_pred)):
        # false negatives
        if y_pred[i] == 0 and y_true[i] == 1:
            fn_file.write('\n{:s}\n'.format(sampled_filenames[i]))
            fn_file.write('Script {:d}\n'.format(i))
            print_command(data[i], int_to_char_dict, fn_file)
        
        # true positives
        if y_pred[i] == 1 and y_true[i] == 1:
            tp_file.write('\n{:s}\n'.format(sampled_filenames[i]))
            tp_file.write('Script {:d}\n'.format(i))
            print_command(data[i], int_to_char_dict, tp_file)
    '''

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('\taccuracy: {:f}'.format(acc))
    print('\tf1: {:f}'.format(f1))
    print('\tprecision: {:f}'.format(prec))
    print('\trecall: {:f}'.format(recall))
    print('\tconfusion matrix (tn, fp, fn, tp): {:d}, {:d}, {:d}, {:d}'.format(tn, fp, fn, tp))
elif args.run:
    # run classification on test-scripts directory
    TEST_DIR = '../test-scripts/'
    for ffile in os.listdir(TEST_DIR):
        TENSOR_LENGTH = 1024
        char_dict = torch.load('../data/prep/char_dict.pth')
        try:
            ps_path = ffile
            ps_file = open(TEST_DIR + ps_path, 'rb')
            ps_tensor = torch.zeros(len(char_dict) + 1, TENSOR_LENGTH) # + 1 for case bit
            tensor_len = min(os.path.getsize(TEST_DIR + ps_path), TENSOR_LENGTH)

            for i in range(tensor_len):
                byte = ps_file.read(1)
                try:
                    byte_char = byte.decode('utf-8')
                except:
                    continue # invalid char
                # check for uppercase
                lower_char = byte_char.lower()
                if byte_char.isupper() and lower_char in char_dict:
                    ps_tensor[len(char_dict)][i] = 1
                    byte_char = lower_char
                # check if byte is most frequent
                if byte_char in char_dict:
                    ps_tensor[char_dict[byte_char]][i] = 1
            ps_file.close()
            # run input through model
            ps_tensor = ps_tensor.view(1, ps_tensor.shape[0], ps_tensor.shape[1]).to(device)
            output = model(ps_tensor)
            # 1 is positive obfuscated, 0 is negative non-obfuscated
            print(ffile + ' output:', int(torch.argmax(output[0])))
        except Exception as e:
            print(e)
else:
    # train model
    print(len(train_loader))
    for i in range(epoch, EPOCHS):
        print('epoch', i)
        # run training
        model.train()
        avg_loss = 0.
        y_true = []
        y_pred = []
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = Variable(data.type(torch.float).to(device)), \
                            Variable(label.type(torch.float).to(device))
            output = model(data)
            loss = mse(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.data
            y_pred += list(torch.max(output, dim=1).indices.cpu().numpy())
            y_true += list(torch.max(label, dim=1).indices.cpu().numpy())

            if batch_idx % int(len(train_loader) / 5) == 0:
                print('\t\tbatch {:d}: {:f}'.format(batch_idx, loss.data))

        # eval model
        # eval_model('train', model, train_loader, len(train_data), mse)
        avg_loss /= len(train_loader)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print('\ttrain loss: {:f}'.format(avg_loss))
        print('\ttrain accuracy: {:f}'.format(acc))
        print('\ttrain f1: {:f}'.format(f1))
        print('\ttrain precision: {:f}'.format(prec))
        print('\ttrain recall: {:f}'.format(recall))
        print('\ttrain confusion matrix (tn, fp, fn, tp): {:d}, {:d}, {:d}, {:d}'.format(tn, fp, fn, tp))

        _, val_f1 = eval_model('val', model, val_loader, len(val_data), mse)

        # save model every epoch if better val f1
        if val_f1 > best_val_f1:
            print('saving this best checkpoint')
            best_val_f1 = val_f1
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1
            }, model_file)
