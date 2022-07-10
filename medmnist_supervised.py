from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy

import medmnist
from medmnist import INFO, Evaluator

import os
from torchvision.utils import save_image
import matplotlib.image

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10
lr = 0.001
weight_decay = 0.001
number_of_res_blocks = 5

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
task

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

number_of_labeled = 9000
labeled_indeces = [i for i in range(len(train_dataset))]
random.shuffle(labeled_indeces)
labeled_indeces = labeled_indeces[0:number_of_labeled]
unlabeled_indeces = np.delete(range(len(train_dataset)), labeled_indeces)


t = transforms.ToTensor()
train_dataset = [(t(train_dataset[i][0]), train_dataset[i][1]) for i in labeled_indeces]
val_dataset = [(t(val_dataset[i][0]), val_dataset[i][1]) for i in range(len(val_dataset))]
test_dataset = [(t(test_dataset[i][0]), test_dataset[i][1]) for i in range(len(test_dataset))]

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader= data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#val_dataset = torchvision.datasets.ImageFolder(val_dir, data_transform)
#test_dataset = torchvision.datasets.ImageFolder(test_dir, data_transform)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader= data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()      
        self.info = 'in ' +  str(in_channels) + ', out ' + str(out_channels) + ', stride ' + str(stride)
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, 0, out_channels - in_channels), mode="constant", value=0))
        else:
            self.skip = nn.Sequential()
 
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        bn1 = nn.BatchNorm2d(out_channels)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        bn2 = nn.BatchNorm2d(out_channels)
        self.l1 = nn.Sequential(conv1, bn1)
        self.l2 = nn.Sequential(conv2, bn2)

    def forward(self, input):
        skip = self.skip(input)
        x = self.l1(input)
        x = F.relu(x)
        x = self.l2(x)
        return F.relu(x + skip)

class ResidualStack(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        first = [ResidualBlock(in_channels, out_channels, stride)]
        rest = [ResidualBlock(out_channels, out_channels) for i in range(num_blocks - 1)]
        self.modules_list = nn.Sequential(*(first + rest))
        
    def forward(self, input):
        return self.modules_list(input)

custom_res_net = nn.Sequential(
          nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding='same', bias=False),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          ResidualStack(16, 16, 1, number_of_res_blocks),
          ResidualStack(16, 32, 2, number_of_res_blocks),
          ResidualStack(32, 64, 2, number_of_res_blocks),
          nn.AdaptiveAvgPool2d(1),
          Lambda(lambda x: x.squeeze()),
          nn.Linear(64, n_classes)
      )

def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def run_epoch(model, optimizer, dataloader, criterion):
  
    model.train()

    epoch_loss = 0.0
    y_score = torch.tensor([])

    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)

        else:
            targets = targets.squeeze().long()

        loss = criterion(outputs, targets)

        outputs = outputs.cpu()
        y_score = torch.cat((y_score, outputs), 0)
        
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    y_score = y_score.detach().numpy()

    evaluator = Evaluator(data_flag, 'train')
    metrics = evaluator.evaluate(y_score)

    epoch_loss /= len(train_loader.dataset)
    return epoch_loss, metrics

def fit(model, optimizer, lr_scheduler, task):

    if task == "multi-label, binary-class":
      criterion = nn.BCEWithLogitsLoss()
    else:
      criterion = nn.CrossEntropyLoss()
    
    _ = model.to(device)
    _ = model.apply(initialize_weight)
    
    best_acc = 0.0
    curr_patience = 0.0
    min_val_loss = 1.0
    best_model_weights = None

    for epoch in range(NUM_EPOCHS):

      train_loss, train_metrics = run_epoch(model, optimizer, train_loader, criterion)
      lr_scheduler.step()
      print(f"Epoch {epoch + 1: >3}/{NUM_EPOCHS}, train loss: {train_loss:.2e}")
      
      val_loss, val_metrics = test(model, val_loader, 'val', criterion)
      print(f"Epoch {epoch + 1: >3}/{NUM_EPOCHS}, val loss: {val_loss:.2e}, val acc: {val_metrics[1]}")

      if val_loss >= min_val_loss:
        curr_patience += 1
        if curr_patience == PATIENCE:
          break
      else:
        curr_patience = 0
        min_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())

      model.load_state_dict(best_model_weights)

def test(model, data_loader, mode, criterion):
    model.eval()
    y_score = torch.tensor([]).to(device)
    acc_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                targets = targets.float().resize_(len(targets), 1)

            acc_loss += loss.item()
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        
        evaluator = Evaluator(data_flag, mode)
        metrics = evaluator.evaluate(y_score)

        acc_loss /= len(train_loader.dataset)
    
    return acc_loss, metrics

optimizer = optim.Adam(custom_res_net.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

if task == 'multi-label, binary-class':
  criterion = nn.BCEWithLogitsLoss()
else:
  criterion = nn.CrossEntropyLoss()


res_net_18 = nn.Sequential(
          nn.Conv2d(n_channels, 64, kernel_size=7, stride=1, padding='same', bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          ResidualStack(64, 64, 1, 6),
          ResidualStack(64, 128, 2, 8),
          ResidualStack(128, 256, 2, 12),
          ResidualStack(256, 512, 1, 6),
          nn.AdaptiveAvgPool2d(1),
          Lambda(lambda x: x.squeeze()),
          nn.Linear(512, n_classes)
      )

fit(res_net_18, optimizer, lr_scheduler, task)

print("****************** SUPERVISED:", test(res_net_18, test_loader, 'test', criterion))