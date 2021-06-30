import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from models import *

DATA_DIR = 'data/processed_tensors/'
EPOCHS = 100
BATCH_SIZE = 128

model = ShallowCNN()
model_file = 'models/cnn-shallow-conv-1-fc-2048-1024.pth'
cuda_device = 0

# set random seed for reproducibility
torch.manual_seed(42)

class ScriptDataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.x = data['x'].to(device)
        self.y = data['y'].to(device)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

parser = argparse.ArgumentParser(description='obfuscation detection train file')
parser.add_argument('--reset', help='start over training', action='store_true')
parser.add_argument('--eval', help='eval model', action='store_true')
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

model_file = 'models/' + args.model_file

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
train_data = ScriptDataset(torch.load(DATA_DIR + 'train_data.pth'), device)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data = ScriptDataset(torch.load(DATA_DIR + 'val_data.pth'), device)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
print('loaded data:', len(train_data), len(val_data))

best_val_acc = 0.
if args.eval or not args.reset:
    # load checkpoint if eval or not retraining
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    best_val_acc = val_acc
    print('loaded {:s} on epoch {:d} with val acc {:f}'.format(model_file, epoch, val_acc))

def eval_model(dataset_name, model, data_loader, num_data, loss_fn):
    model.eval()
    avg_loss = 0.
    num_batches = 0
    y_true = []
    y_pred = []
    for _, (data, label) in enumerate(data_loader):
        # run model
        data, label = Variable(data), Variable(label)
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
    return avg_loss, acc

if args.eval:
    # eval model
    eval_model('train', model, train_loader, len(train_data), mse)
    eval_model('val', model, val_loader, len(val_data), mse)
else:
    # train model
    for i in range(epoch, EPOCHS):
        print('epoch', i)
        # run training
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = Variable(data), Variable(label)
            output = model(data)
            loss = mse(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print('\t\tbatch {:d}: {:f}'.format(batch_idx, loss.data))

        # eval model
        eval_model('train', model, train_loader, len(train_data), mse)
        _, val_acc = eval_model('val', model, val_loader, len(val_data), mse)

        # save model every epoch if better val acc
        if val_acc > best_val_acc:
            print('saving this best checkpoint')
            best_val_acc = val_acc
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, model_file)
