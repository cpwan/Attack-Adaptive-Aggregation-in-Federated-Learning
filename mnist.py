from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataloader import *
import pickle

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def basic_loader(num_clients,loader_type):
    dataset = datasets.MNIST('./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    return loader_type(num_clients,dataset)

def train_dataloader(num_clients,loader_type='iid' ,store=True,path='./data/loader.pk'):
    assert loader_type in ['iid','byLabel','dirichlet'], 'Loader has to be either \'iid\' or \'non_overlap_label \''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

        
    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('Loader not found, initializing one')
            loader = basic_loader(num_clients,loader_type)
    else:
        print('Initialize a data loader')
        loader = basic_loader(num_clients,loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)   
    
    return loader
    

def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=test_batch_size, shuffle=True)
    return test_loader

if __name__ == '__main__':
    print("#Initialize a network")
    net = Net()
    batch_size = 100
    y = net((torch.randn(batch_size,1,28,28)))
    print(f"Output shape of the network with batchsize {batch_size}:\t {y.size()}")
    
    print("\n#Initialize dataloaders")
    loader_types = ['iid','byLabel','dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(10,loader_types[i],store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")
    
    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0]
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
