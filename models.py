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

# best acc: 98.67% train, 84.56% val
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(73728, 1024),
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
            nn.Linear(73728, 4096),
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
            nn.Conv2d(1, 128, kernel_size=(72, 3), stride=1), # 72 for num of chars
            nn.ReLU(),
            View((-1, 128, 1022)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14620, 2048),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.9),
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
            nn.Conv2d(1, 256, kernel_size=(72, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 256, 339)),
            # conv2
            nn.Conv2d(1, 256, kernel_size=(256, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=3),
            View((-1, 1, 256, 111)),
            # conv3
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            View((-1, 1, 256, 109)),
            # conv4
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            View((-1, 1, 256, 107)),
            # conv5
            nn.Conv2d(1, 256, kernel_size=(256, 3), stride=1),
            nn.ReLU(),
            View((-1, 1, 256, 105)),
            # conv6
            nn.Conv2d(1, 256, kernel_size=(256,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # fc1
            nn.Linear(8704, 1024),
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
