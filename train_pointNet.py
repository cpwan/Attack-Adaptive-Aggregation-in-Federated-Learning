#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Agg_dataset(torch.utils.data.Dataset):
    '''
        denote n be the number of clients,
        each entry of dataset is a 2-tuple of (weight delta, labels):= (1 x n tensor, 1 x n tensor)
        honest clients are labeled 1, malicious clients are labeled 0
    '''
    def __init__(self,path,attacks):
        super(Agg_dataset).__init__() 
        data=torch.load(path,map_location='cpu')
        data_tensors=torch.cat([data[param] for param in data],0)
        self.data=data_tensors
        self.label=attacks
        label=attacks/torch.sum(attacks)
        self.center=torch.sum(data_tensors*label,1).view(-1,1)
        self.num_clients=attacks.shape[0]
        self.n=nsamples
        dimension=vector_dimension
        self.indexes=torch.randint(self.data.shape[0],(self.n,vector_dimension))
        
    def __getitem__(self, index):
#         

        data_out=self.data[self.indexes[index]]
        label_out=self.label
        center_out=self.center[self.indexes[index]]
        perm=torch.randperm(self.num_clients)
#         data_sorted=torch.sort(data_out)
#         perm=data_sorted[1]
        data_out_shuffled=torch.index_select(data_out, -1, perm)

        return data_out_shuffled, [label_out[perm],center_out]
#         return data_out,[label_out,center_out]
    def __len__(self):
#         return self.data.shape[0]//100
        return self.n


# In[4]:

def get_concat_loader():
    datasets=[]
    for epoch in range(3):
        for attack in attacks:
            label=torch.ones(10)
            for i in attacker_list_labelflipping[attack]:
                label[i]=0
            for i in attacker_list_omniscient[attack]:
                label[i]=0
            path=f'./AggData/{attack}/FedAvg_{epoch}.pt'
            dataset=Agg_dataset(path,label)
            datasets.append(dataset)
    dataset=torch.utils.data.dataset.ConcatDataset(datasets)
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, test_loader


def get_loader(attack):
    label=torch.ones(10)
    for i in attacker_list_labelflipping[attack]:
        label[i]=0
    for i in attacker_list_omniscient[attack]:
        label[i]=0
    path=f'./AggData/{attack}/FedAvg_9.pt'
    dataset=Agg_dataset(path,label)
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, test_loader


# In[5]:


def train(net,train_loader,criterion,optimizer,device, on=0):
    net.to(device)
    for idx, (data,target) in enumerate(train_loader):
        data = data.to(device)
        target = target[on].to(device)
        optimizer.zero_grad()   
        output = net(data)
        loss = criterion(output[on], target)
        loss.backward()
        optimizer.step()
    
def test(net,test_loader,device,message_prefix):
    net.to(device)
    accuracy = 0
    accuracy_binary = 0
    accuracy_mean = 0
    accuracy_median = 0
    count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target[1].to(device)
            outputs = net(data)
            accuracy+=F.l1_loss(outputs[1], target)
            accuracy_binary+=F.l1_loss(outputs[2], target)
            accuracy_mean+=F.l1_loss(data.mean(-1).unsqueeze(-1), target)
            accuracy_median+=F.l1_loss(data.median(-1)[0].unsqueeze(-1), target)
            count+=len(data)
    print('%s: \t%.4E \t %.4E \t%.4E \t%.4E' % (message_prefix,accuracy/count, accuracy_binary/count, accuracy_mean/count,accuracy_median/count ))
    return accuracy/count, accuracy_binary/count, accuracy_mean/count, accuracy_median/count


# In[11]:
if __name__=="__main__":
    
    
    torch.manual_seed(0)
    num_epoch=200
    batch_size=32
    device='cuda'
    data_root='./data'
    vector_dimension=1
    nsamples=10000

    import allocateGPU
    allocateGPU.allocate_gpu()
    
    from pointNet import PointNet
    import torch.optim as optim



    # In[2]:

    attacks=['No-Attacks',
             'Omniscient1',
             'Omniscient2',
             'Omniscient3',
             'Omniscient4',
             'Omniscient5',
             'Label-Flipping1',
             'Label-Flipping2',
             'Label-Flipping3',
             'Label-Flipping4',
             'Label-Flipping5']
    
    attacker_list_labelflipping={'No-Attacks':[],
                                 'Omniscient1':[],
                                 'Omniscient2':[],
                                 'Omniscient3':[],
                                 'Omniscient4':[],
                                 'Omniscient5':[],
                                 'Label-Flipping1':[0],
                                 'Label-Flipping2':[1,2],
                                 'Label-Flipping3':[3,6,9],
                                 'Label-Flipping4':[1,3,5,7],
                                 'Label-Flipping5':[0,2,4,6,8]
                                }
    attacker_list_omniscient =  {'No-Attacks':[],
                                 'Omniscient1':[0],
                                 'Omniscient2':[1,2],
                                 'Omniscient3':[3,6,9],
                                 'Omniscient4':[1,3,5,7],
                                 'Omniscient5':[0,2,4,6,8],
                                 'Label-Flipping1':[],
                                 'Label-Flipping2':[],
                                 'Label-Flipping3':[],
                                 'Label-Flipping4':[],
                                 'Label-Flipping5':[]}

    from tensorboardX import SummaryWriter
    def write(name,scalar):
        writer=SummaryWriter(f'./agg_logs/{name}')
        writer.add_scalar('l1 loss', scalar, 0)
        writer.close()


    # In[12]:


    accuracy_list={}


    # In[13]:


    train_loader, test_loader=get_concat_loader()


    for attack in attacks[:]:
        for criterion in [torch.nn.BCELoss()]:
            mode=0
            for lr in [0.01]:

                net_ptnet=PointNet(vector_dimension,10)

                for net in [net_ptnet]:

                    training_alias=f'mulitple_attack/{net.__class__.__name__}/{criterion.__class__.__name__}/lr_{lr}'
                    if training_alias in accuracy_list:
                        continue

                    print('Start training of %s'%training_alias)
                    optimizer = optim.Adam(net.parameters(), lr=lr)
                    print('L1 loss:\tmodel \tmodel (binarized) \tmean \tmedian ')
                    for epoch in range(num_epoch):
                        train(net,train_loader,criterion,optimizer,device,mode)
                        score=test(net,test_loader,device,f'Epoch {epoch}')
                        if epoch%20==19:
                            torch.save(net.state_dict(),f'pointNetAR_big_{epoch}.pt')
                        if epoch%10==9:
                            #use another set of data once in a while
                            train_loader, test_loader = get_concat_loader()
                        
                    write(training_alias+'/weighted',score[0])
                    write(training_alias+'/binary',score[1])
                    accuracy_list[training_alias]=score[0].item()
        accuracy_list[f'mulitple_attackmean']=score[1].item()
        write(f'mulitple_attack/mean',score[1])
        accuracy_list[f'mulitple_attack/median']=score[2].item()
        write(f'mulitple_attack/median',score[1])



