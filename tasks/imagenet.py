from __future__ import print_function
import sys, os 
sys.path.append(os.getcwd()) # for running this file directly from the parent directory "python ./tasks/imagenet.py"
import pickle

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from dataloader import *
from torchvision.models.resnet import resnet18

path_to_tiny_imagenet="./data/imagenet/tiny-imagenet-200/"
train_on_testset=False

# def Net():
#     num_classes = 200
#     model = EfficientNet.from_pretrained('efficientnet-b0')
#     n = model._fc.in_features
#     model._fc = nn.Linear(n, num_classes)
#     return model
def Net():
    num_classes = 200
#     model = resnet18(pretrained=True)
    model = resnet18(pretrained=False)
    n = model.fc.in_features
    model.fc = nn.Linear(n, num_classes)
    return model

def getDataset():
    transform=transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomResizedCrop(224),
#             transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    if train_on_testset:
        phase='val'
    else:
        phase='train'
    dataset = datasets.ImageFolder(
        root=path_to_tiny_imagenet+phase, 
        transform=transform)
    return dataset


def basic_loader(num_clients, loader_type, batchsize):
    dataset = getDataset()
    return loader_type(num_clients, dataset, bsz=batchsize)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk', batchsize=128):
    assert loader_type in ['iid', 'byLabel',
                           'dirichlet'], 'Loader has to be one of the  \'iid\',\'byLabel\',\'dirichlet\''
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
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_type, batchsize)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type, batchsize)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    
    transform=transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    test_loader = torch.utils.data.DataLoader(
            dataset = datasets.ImageFolder(
                root=path_to_tiny_imagenet+"val",
                transform=transform),
            batch_size=test_batch_size, shuffle=False)
    return test_loader


if __name__ == '__main__':
    from torchsummary import summary

    print("#Initialize a network")
    net = Net()
    summary(net.cuda(), (3, 64,64))
    print(torch.cuda.memory_summary(abbreviated=True))
    
    sdict=net.state_dict()
    for param in sdict:
        if 'Float' not in sdict[param].type():
            print(f'{param}\t{sdict[param].shape}\t{sdict[param].type()}')
    
    
    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(100, loader_types[i], store=False)
        if loader_types[i]=='dirichlet':
            print("alpha",loader.alpha)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")
        print(f"Shape of a minibatch: {next(iter(loader[i]))[0].shape}")
    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
    print(torch.cuda.memory_summary(abbreviated=True))

