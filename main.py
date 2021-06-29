import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import MLP

MODEL_FILE = 'models/mlp-shallow-1024-512.pth'
DATA_DIR = 'data/processed_tensors/'
EPOCHS = 16
BATCH_SIZE = 128

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
args = parser.parse_args()

device = torch.device('cpu')
if torch.cuda.is_available():
    print('using CUDA')
    device = torch.device('cuda')

if args.eval and args.reset:
    print('cannot eval without loading model')
    exit(1)

# init model
model = MLP()
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

if args.eval or not args.reset:
    # load checkpoint if eval or not retraining
    checkpoint = torch.load(MODEL_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print('loaded {:s} on epoch {:d} with val acc {:f}'.format(MODEL_FILE, epoch, checkpoint['val_acc']))

def eval_model(dataset_name, model, data_loader, num_data, loss_fn):
    model.eval()
    avg_loss = 0.
    acc = 0.
    num_batches = 0
    for _, (data, label) in enumerate(data_loader):
        # run model
        data, label = Variable(data), Variable(label)
        output = model(data)

        # calculate loss
        loss = loss_fn(output, label)
        avg_loss += loss.data
        num_batches += 1
        # calculate accuracy
        output_labels = torch.max(output, dim=1).indices
        label = torch.max(label, dim=1).indices
        acc += (output_labels == label).sum()
    avg_loss /= num_batches
    acc /= num_data
    print('\t{:s} loss: {:f}'.format(dataset_name, avg_loss))
    print('\t{:s} acc: {:f}'.format(dataset_name, acc))
    return avg_loss, acc

if args.eval:
    # eval model
    eval_model('train', model, train_loader, len(train_data), mse)
    eval_model('val', model, val_loader, len(val_data), mse)
else:
    # train model
    best_val_acc = 0.
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
            }, MODEL_FILE)
