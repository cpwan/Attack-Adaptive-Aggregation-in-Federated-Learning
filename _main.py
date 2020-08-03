from __future__ import print_function
from random import Random

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter


import mnist
import cifar
from server import Server
from clients import Client
from modules import Net
from allocateGPU import *
from clients_attackers import *

def main(args):
    
    print('#####################')
    print('#####################')
    print('#####################')
    print(f'Gradient Aggregation Rule:\t{args.GAR}\nData distribution:\t{args.loader_type}\nAttacks:\t{args.attacks} ')
    print('#####################')
    print('#####################')
    print('#####################')
    
    torch.manual_seed(args.seed)
    
   
    
    
 
    device='cuda'
    allocate_gpu()
    
    attacks=args.attacks
    
    writer=SummaryWriter(f'./logs/{args.output_folder}/{args.experiment_name}')

    
    if args.dataset=='mnist':
        trainData=mnist.train_dataloader(args.num_clients,loader_type=args.loader_type,path=args.loader_path, store=False)
        testData=mnist.test_dataloader(args.test_batch_size)
        Net=mnist.Net
        criterion=F.nll_loss
    elif args.dataset=='cifar':
        trainData=cifar.train_dataloader(args.num_clients,loader_type=args.loader_type,path=args.loader_path, store=False)
        testData=cifar.test_dataloader(args.test_batch_size)
        Net=cifar.Net
        criterion=F.cross_entropy
    #create server instance
    model0 = Net().to(device)
    server=Server(model0,testData,criterion,device)
    server.set_GAR(args.GAR)
    server.path_to_aggNet=args.path_to_aggNet
    if args.save_model_weights:
        server.isSaveChanges=True
        server.savePath=f'./AggData/{args.loader_type}/{args.dataset}/{args.attacks}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)
        '''
        honest clients are labeled 1, malicious clients are labeled 0
        '''
        label=torch.ones(10)
        for i in args.attacker_list_labelFlipping:
            label[i]=0
        for i in args.attacker_list_labelFlippingDirectional:
            label[i]=0
        for i in args.attacker_list_omniscient:
            label[i]=0
        for i in args.attacker_list_backdoor:
            label[i]=0
        torch.save(label,f'{server.savePath}/label.pt')
    #create clients instance
    
    attacker_list_labelFlipping           =args.attacker_list_labelFlipping
    attacker_list_omniscient              =args.attacker_list_omniscient
    attacker_list_backdoor                =args.attacker_list_backdoor
    attacker_list_labelFlippingDirectional=args.attacker_list_labelFlippingDirectional
    for i in range(args.num_clients):
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        if i in attacker_list_labelFlipping:
            client_i=Attacker_LabelFlipping(i,model,trainData[i],optimizer,criterion,device)
        elif i in attacker_list_labelFlippingDirectional:
            client_i=Attacker_LabelFlippingDirectional(i,model,trainData[i],optimizer,criterion,device)
        elif i in attacker_list_omniscient:
            client_i=Attacker_Omniscient(i,model,trainData[i],optimizer,criterion,device,args.omniscient_scale)
        elif i in attacker_list_backdoor:
            client_i=Attacker_Backdoor(i,model,trainData[i],optimizer,criterion,device)
        else:
            client_i=Client(i,model,trainData[i],optimizer,criterion,device)
        server.attach(client_i)
        
    loss,accuracy=server.test()
    steps=0
    writer.add_scalar('test/loss', loss, steps)
    writer.add_scalar('test/accuracy', accuracy, steps)
    
    if 'BACKDOOR' in args.attacks.upper():
        loss,accuracy=server.test_backdoor()

        writer.add_scalar('test/loss_backdoor', loss, steps)
        writer.add_scalar('test/backdoor_success_rate', accuracy, steps)
    for j in range(args.epochs):        
        steps=j+1
        
        print('\n\n########EPOCH %d ########'%j)
        print('###Model distribution###\n')
        server.distribute()
#         group=Random().sample(range(5),1)
        group=range(args.num_clients)
        server.train(group)
#         server.train_concurrent(group)
        
        loss,accuracy=server.test()
        
        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)
        
        if 'BACKDOOR' in args.attacks.upper():
            loss,accuracy=server.test_backdoor()

            writer.add_scalar('test/loss_backdoor', loss, steps)
            writer.add_scalar('test/backdoor_success_rate', accuracy, steps)
        
        
    writer.close()
  