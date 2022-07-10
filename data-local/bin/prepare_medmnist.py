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
import random
import sys

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

workdir = os.getcwd() + "/data-local/images/"
os.makedirs(workdir, exist_ok=True)

train_dataset = DataClass(split='train', download=download)
val_dataset = DataClass(split='val', download=download)
test_dataset = DataClass(split='test', download=download)


train_dir = os.path.abspath(os.path.join(workdir, 'train'))
val_dir = os.path.abspath(os.path.join(workdir, 'val'))
test_dir = os.path.abspath(os.path.join(workdir, 'test'))

number_of_labeled = int(sys.argv[1])
print("----------number_of_labeled:", number_of_labeled)
number_of_class = 9
"""
labeled_indeces = [i for i in range(len(train_dataset))]
random.shuffle(labeled_indeces)
labeled_indeces = labeled_indeces[0:number_of_labeled]
unlabeled_indeces = np.delete(range(Train_len), labeled_indeces)
"""

label_per_class = number_of_labeled // number_of_class
Train_len = len(train_dataset)
labels = np.array([train_dataset[i][1][0] for i in range(Train_len)])
labeled_indeces = []
unlabeled_idx = np.array(range(len(labels))) 
for i in range(number_of_class): 
    idx = np.where(labels == i)[0] 
    idx = np.random.choice(idx, label_per_class, False) 
    labeled_indeces.append(idx) 
labeled_indeces = np.array(labeled_indeces)

try:
    Train_percentage = int(sys.argv[2])
    unlabeled_indeces = np.random.choice(np.delete(range(Train_len), labeled_indeces), (Train_len * Train_percentage) // 100, False)
except:
    unlabeled_indeces = np.delete(range(Train_len), labeled_indeces)


label_names = info['label']
print(label_names)

def write_image(target_dir, index, x, y):
    subdir = os.path.join(target_dir, str(y[0]))
    name = "{}_{}.png".format(index, y[0])
    os.makedirs(subdir, exist_ok=True)
    t = transforms.ToTensor()
    save_image(t(x), os.path.join(subdir, name))
    return

file_object = open(os.path.join(workdir, "labels.txt"), 'a')

for i in range(len(train_dataset)):
    x, y = train_dataset[i][0], train_dataset[i][1]
    
    write_image(train_dir, i, x, y)
    if i in labeled_indeces:
        file_object.write('{}_{}.png {}\n'.format(i, y[0], y[0]))

file_object.close()

for i in range(len(val_dataset)):
    x, y = val_dataset[i][0], val_dataset[i][1]
    write_image(val_dir, i, x, y)

for i in range(len(test_dataset)):
    x, y = test_dataset[i][0], test_dataset[i][1]
    write_image(test_dir, i, x, y)    
    