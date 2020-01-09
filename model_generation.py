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
from server import Server
from clients import Client
from modules import Net
from allocateGPU import *
from clients_attackers import *


def main(args):
    torch.manual_seed(args.seed)
    
   
    
    
 
    device='cuda'
    attacks=args.attacks
    
    
    trainData=mnist.train_dataloader(args.num_clients,loader_type=args.loader_type,path=args.loader_path)
    testData=mnist.test_dataloader(args.test_batch_size)
    
    #create server instance
    model0 = Net().to(device)
    server=Server(model0,testData,device)
    server.set_GAR(args.GAR)
    server.isSaveChanges=True
    server.savePath=f'./AggData/{args.attacks}'
    
    import os
    os.makedirs(server.savePath, exist_ok=True)
    #create clients instance
    
    attacker_list_labelflipping=args.attacker_list_labelflipping
    attacker_list_omniscient=args.attacker_list_omniscient
    for i in range(args.num_clients):
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        if i in attacker_list_labelflipping:
            client_i=Attacker_LabelFlipping(i,model,trainData[i],optimizer,device)
        elif i in attacker_list_omniscient:
            client_i=Attacker_Omniscient(i,model,trainData[i],optimizer,device)
        else:
            client_i=Client(i,model,trainData[i],optimizer,device)
        server.attach(client_i)
        
    loss,accuracy=server.test()
    steps=0

    for j in range(10):
        print('\n\n########EPOCH %d ########'%j)
        print('###Model distribution###\n')
        server.distribute()
#         group=Random().sample(range(5),1)
        group=range(args.num_clients)
        server.train(group)
        loss,accuracy=server.test()
        steps=j+1

        
if __name__=="__main__":
    allocate_gpu()
    
    class Arguments():
        def __init__(self):
            self.batch_size=64
            self.test_batch_size=1000
            self.epochs=10
            self.lr=0.01
            self.momentum=0.5
            self.no_cuda=False
            self.seed=1
            self.log_interval=10
            self.num_clients=10
            self.output_folder='model'
    #             self.loader_type='non_overlap_label'
    #             self.loader_path='./data/non_overlap_loader.pk'
            self.loader_type='iid'
            self.loader_path='./data/iid_loader.pk'
            self.GAR='median'
            self.attacker_list_labelflipping=[]
            self.attacker_list_omniscient=[]
            self.attacks=''#'Omniscient','labelFlipping'
    args=Arguments()
    
    GAR=['fedavg']#,'median','deepGAR']
    loader=[('iid','./data/iid_loader.pk'),]#('non_overlap_label','./data/non_overlap_loader.pk')]
#     loader=[('non_overlap_label','./data/non_overlap_loader.pk')]

    attacks=['no_attacks','omniscient','label_flipping','omniscient_aggresive']
#     attacks=['Omniscient_Aggresive']
    attacker_list_labelflipping={'no_attacks':[],'omniscient':[],'label_flipping':[0],'omniscient_aggresive':[]}
    attacker_list_omniscient={'no_attacks':[],'omniscient':[0],'label_flipping':[],'omniscient_aggresive':[0]}
    for attack in attacks:
        for (loader_type,loader_path) in loader:
            for gar in GAR:              
                args.GAR=gar
                args.loader_type=loader_type
                args.loader_path=loader_path
                args.attacks=attack
                args.attacker_list_labelflipping=attacker_list_labelflipping[attack]
                args.attacker_list_omniscient=attacker_list_omniscient[attack]
                
                print('#####################')
                print('#####################')
                print('#####################')
                print(f'Gradient Aggregation Rule:\t{gar}\nData distribution:\t{loader_type}\nAttacks:\t{attack} ')
                print('#####################')
                print('#####################')
                print('#####################')
                main(args)


