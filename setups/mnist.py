from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataloader import labelLoader,iidLoader
import pickle

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def basic_loader(num_clients,loader_type=labelLoader ):
    dataset=datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    return loader_type(num_clients,dataset)

def train_dataloader(num_clients,loader_type=labelLoader ,store=True,path='./data/loader.pk'):
    assert loader_type in ['iid','non_overlap_label'], 'Loader has to be either \'iid\' or \'non_overlap_label \''
    if loader_type=='iid':
        loader_type=iidLoader
    elif loader_type=='non_overlap_label':
        loader_type=labelLoader

        
    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('loader not found, initialize one')
            loader=basic_loader(num_clients,loader_type)
    else:
        print('initialize a data loader')
        loader=basic_loader(num_clients,loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)   
    
    return loader
    

def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)
    return test_loader