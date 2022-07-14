import gc
import pickle
import logging

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import os
import shutil

import numpy as np
import torch

from Datasets.data import NO_LABEL
from misc.utils import *
#from tensorboardX import SummaryWriter
import datetime
from parameters import get_parameters
import models

from misc import ramps
from Datasets import data as data2
from models__ import losses

import torchvision.transforms as transforms


import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets
import torch.utils.data as dt

import copy

logger = logging.getLogger(__name__)

args_consistency = 7
args_consistency_rampup = 5
args_ema_decay = 0.999

def update_ema_variables(model, ema_model, global_step):
    alpha = min(1 - 1 / (global_step + 1), args_ema_decay)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        #ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha) #add_(1 - alpha, param.data)
        ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data

def get_current_consistency_weight(epoch):
    return args_consistency * ramps.sigmoid_rampup(epoch, args_consistency_rampup)

consistency_criterion = losses.softmax_mse_loss

class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None
        self.__teacher_model = None
        self.indices = local_data
        self.global_step = 0

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model
    
    @property
    def teacher_model(self):
        """Local model getter for parameter aggregation."""
        return self.__teacher_model

    @teacher_model.setter
    def teacher_model(self, teacher_model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__teacher_model = teacher_model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        self.batch_size = client_config["batch_size"]
        ratio = 0.1
        idx = np.arange(len(self.data))
        np.random.shuffle(idx)

        labeled_idxs = idx[:int(len(idx) * ratio)]
        unlabeled_idxs = idx[int(len(idx) * ratio):]
        self.labeled_idxs = labeled_idxs
        self.unlabeled_idxs = unlabeled_idxs
        batch_sampler2 = data2.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, self.batch_size, self.batch_size // 2)

        """Set up common configuration of each client; called by center server."""
        X, y = self.data.tensors
        #print(X.shape, y.shape)
        dataset = dt.TensorDataset(X, y)
        #self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler2,
                                               num_workers=2,
                                               pin_memory=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)
        self.teacher_model.train()
        self.teacher_model.to(self.device)

        loss_f = eval(self.criterion)()
      
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for i, (data, labels) in enumerate(self.dataloader):#(data, unlabeled_data), labels in self.dataloader: # (
                #((data, ema_input), target)
                ema_input, data = data[:self.batch_size // 2], data[self.batch_size // 2:]
                labels = labels[self.batch_size // 2:] #ema_labels wont be used for the semisupervised experiment

                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                outputs = self.model(data)

                #### EMA input calculation
                with torch.no_grad():
                  ema_input_var = torch.autograd.Variable(ema_input).cuda()
                #ema_input_var = ema_input_var.cuda()
                ema_logit = self.teacher_model(ema_input_var)#, ema_input=True)
                #ema_logit = ema_model_out
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                #consistency_weight =
                
                #### EMA input calculation

                outputs2 = self.model(ema_input_var)
                outputs2 = Variable(outputs2.detach().data, requires_grad=False)
                
                semisupervised = False
                if semisupervised:
                    loss = loss_f(outputs, labels) + get_current_consistency_weight(e) \
                        * consistency_criterion(outputs2, ema_logit)
                else:
                    loss = loss_f(outputs, labels)

                loss.backward()
                optimizer.step()

                self.global_step += 1

                if semisupervised:
                    update_ema_variables(self.model, self.teacher_model, self.global_step) 
                else:
                    self.teacher_model = copy.deepcopy(self.model)

                if self.device == "cuda": torch.cuda.empty_cache()               
        self.model.to("cpu")
        self.teacher_model.to("cpu")

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.teacher_model.eval()
        self.model.to(self.device)
        self.teacher_model.to(self.device)

        test_loss1, correct1 = 0, 0
        test_loss2, correct2 = 0, 0
        total_size = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.dataloader):#(data, unlabeled_data), labels in self.dataloader:
                #ema_input, data = data[:self.batch_size // 2], data[self.batch_size // 2:]
                #labels = labels[self.batch_size // 2:]
                
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs1 = self.model(data)
                test_loss1 += eval(self.criterion)()(outputs1, labels).item()
                
                predicted1 = outputs1.argmax(dim=1, keepdim=True)
                correct1 += predicted1.eq(labels.view_as(predicted1)).sum().item()

                outputs2 = self.teacher_model(data)
                test_loss2 += eval(self.criterion)()(outputs2, labels).item()
                
                predicted2 = outputs2.argmax(dim=1, keepdim=True)
                correct2 += predicted2.eq(labels.view_as(predicted2)).sum().item()

                total_size += data.shape[0]

                if self.device == "cuda": torch.cuda.empty_cache()

        test_loss1 = test_loss1 / len(self.dataloader)
        test_accuracy1 = correct1 / len(self.data)

        test_loss2 = test_loss2 / len(self.dataloader)
        test_accuracy2 = correct2 / len(self.data)

        
        #print("{} Student model Loss: {} and Accuracy: {}".format(test_loss, test_accuracy))

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Student Test loss: {test_loss1:.4f}\
            \n\t=> Student Test accuracy: {100. * test_accuracy1:.2f}%\
            \n\t=> Teacher Test loss: {test_loss2:.4f}\
            \n\t=> Teacher Test accuracy: {100. * test_accuracy2:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss2, test_accuracy2
