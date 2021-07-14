import torch
import torch.nn as nn


class Shape(nn.Module):
    def __init__(self):
        super(Shape, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

###
# start ResNet classes

class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = self.tanh(x[:, :int(x.shape[1]/2)])
        s = self.sigmoid(x[:, int(x.shape[1]/2):])
        return t * s

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearNorm, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain('linear'))
    
    def forward(self, x):
        x = self.flatten(x) # flatten b/c it's too large for fc right now?
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        nn.init.xavier_normal_(
            self.conv.weight,
            gain=nn.init.calculate_gain('tanh'))

    def forward(self, x):
        return self.conv(x)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        NUM_LAYERS = 3
        NUM_FILTERS = 512
        input_size = 71
        convolutions_char = []
        self.num_filters = NUM_FILTERS
        for _ in range(NUM_LAYERS):
            conv_layer = nn.Sequential(
                ConvNorm(input_size, NUM_FILTERS,
                            kernel_size=5, stride=1,
                            padding=2, dilation=1),
                nn.BatchNorm1d(NUM_FILTERS)
            )
            convolutions_char.append(conv_layer)
            input_size = NUM_FILTERS // 2
        self.convolutions_char = nn.ModuleList(convolutions_char)
        self.pre_out = LinearNorm(NUM_FILTERS // 2, 2)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        half = self.num_filters // 2
        res = None
        skip = None
        for i in range(len(self.convolutions_char)):
            conv = self.convolutions_char[i]
            drop = True
            if i >= len(self.convolutions_char) - 1:
                drop = False
            if skip is not None:
                x = x + skip
            conv_out = conv(x)
            tmp = torch.tanh(conv_out[:, :half, :]) * torch.sigmoid(conv_out[:, half:, :])
            if res is None:
                res = tmp
            else:
                res = res + tmp
            skip = tmp
            x = torch.dropout(tmp, 0.1, drop)
        x = x + res
        x = x.permute(0, 2, 1)
        pre = torch.sum(x, dim=1, dtype=torch.float)
        pre /= 4096
        return torch.softmax(self.pre_out(pre), dim=1)

# end ResNet classes
###

# best acc: 98.67% train, 84.56% val
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72704, 1024),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

# best acc so far: 92.11% train, 82.78%, taking forever to train
class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72704, 4096),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Hendler (2018)
# best acc: 96.94% train, 89.68% val
class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(71, 3), stride=1), # 71 for num of chars
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(43520, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class GatedCNN(nn.Module):
    def __init__(self):
        super(GatedCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(71, 3), stride=1), # 71 for num of chars
            GatedActivation(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14620, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Zhang (2015)
# best acc: not run yet
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 16, kernel_size=(71, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 16, 339)),
            # conv2
            nn.Conv2d(1, 16, kernel_size=(16, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 16, 111)),
            # conv3
            nn.Conv2d(1, 16, kernel_size=(16, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 16, 36)),
            # conv4
            nn.Conv2d(1, 16, kernel_size=(16, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 16, 11)),
            # conv5
            nn.Conv2d(1, 16, kernel_size=(16, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # output
            nn.Linear(48, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 128, kernel_size=(71, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 128, 339)),
            # conv2
            nn.Conv2d(1, 128, kernel_size=(128,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(14336, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # fc2
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # output
            nn.Linear(1024, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Conv2Gated(nn.Module):
    def __init__(self):
        super(Conv2Gated, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 256, kernel_size=(71, 7), stride=1),
            GatedActivation(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 128, 339)),
            # conv2
            nn.Conv2d(1, 256, kernel_size=(128,3), stride=1),
            GatedActivation(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(14336, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # fc2
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # output
            nn.Linear(1024, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class LargeCNN2(nn.Module):
    def __init__(self):
        super(LargeCNN2, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 1024, kernel_size=(71, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 1024, 339)),
            # conv2
            nn.Conv2d(1, 1024, kernel_size=(1024,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(114688, 2048),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # fc2
            nn.Linear(2048, 2048),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # output
            nn.Linear(2048, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 128, kernel_size=(71, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 128, 339)),
            # conv2
            nn.Conv2d(1, 128, kernel_size=(128, 3), stride=1),
            nn.ReLU(),
            View((-1, 1, 128, 337)),
            # conv3
            nn.Conv2d(1, 128, kernel_size=(128,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(14208, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # fc2
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # output
            nn.Linear(1024, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 256, kernel_size=(71, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 256, 339)),
            # conv2
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
            View((-1, 1, 256, 112)),
            # conv3
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            View((-1, 1, 256, 110)),
            # conv4
            nn.Conv2d(1, 256, kernel_size=(256,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(9216, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # fc2
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # output
            nn.Linear(1024, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Conv5(nn.Module):
    def __init__(self):
        super(Conv5, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(1, 256, kernel_size=(71, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 256, 339)),
            # conv2
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
            View((-1, 1, 256, 112)),
            # conv3
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            View((-1, 1, 256, 110)),
            # conv4
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            View((-1, 1, 256, 108)),
            # conv5
            nn.Conv2d(1, 256, kernel_size=(256,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(8960, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # fc2
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # output
            nn.Linear(1024, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(71, 256, batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(262144, 256),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # fc2
            nn.Linear(256, 256),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            # output
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )
    
    # def forward(self, x, hn, cn):
    def forward(self, x):
        # x = self.lstm(x, (hn, cn))
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class SmallLSTM(nn.Module):
    def __init__(self):
        super(SmallLSTM, self).__init__()
        self.lstm = nn.LSTM(71, 128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(131072, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # fc2
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # output
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )
    
    # def forward(self, x, hn, cn):
    def forward(self, x):
        # x = self.lstm(x, (hn, cn))
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class LargeLSTM(nn.Module):
    def __init__(self):
        super(LargeLSTM, self).__init__()
        self.lstm = nn.LSTM(71, 512, batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(524288, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # fc2
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # output
            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )
    
    # def forward(self, x, hn, cn):
    def forward(self, x):
        # x = self.lstm(x, (hn, cn))
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
