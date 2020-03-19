from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet18
from dataloader import labelLoader,iidLoader
import pickle
from setups.resnet import ResNet18

def Net():
    model = resnet18(pretrained=True)
    n=model.fc.in_features
    model.fc=nn.Linear(n,10)
    return model


def basic_loader(num_clients,loader_type=labelLoader ):
    dataset=datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5 ))
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
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5 ))
                   ])),
    batch_size=test_batch_size, shuffle=True)
    return test_loader