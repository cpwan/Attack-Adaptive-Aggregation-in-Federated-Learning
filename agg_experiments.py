#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.datasets as dsets
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
num_epoch=20
batch_size=32
device='cuda'
data_root='./data'

import allocateGPU
allocateGPU.allocate_gpu()


# In[2]:


attacks=['no_attacks','omniscient','label_flipping','omniscient_aggresive']
#     attacks=['Omniscient_Aggresive']
attacker_list_labelflipping={'no_attacks':[],'omniscient':[],'label_flipping':[0],'omniscient_aggresive':[]}
attacker_list_omniscient={'no_attacks':[],'omniscient':[0],'label_flipping':[],'omniscient_aggresive':[0]}


# In[14]:


class Agg_dataset(torch.utils.data.Dataset):
    '''
        denote n be the number of clients,
        each entry of dataset is a 2-tuple of (weight delta, labels):= (1 x n tensor, 1 x n tensor)
        honest clients are labeled 1, malicious clients are labeled 0
    '''
    def __init__(self,path,attacks):
        super(Agg_dataset).__init__() 
        data=torch.load(path)
        data_tensors=torch.cat([data[param] for param in data],0)
        self.data=data_tensors.cuda()
        self.label=attacks/torch.sum(attacks)
        self.label=self.label.cuda()
        self.center=torch.sum(data_tensors*self.label,1).view(-1,1)
        self.num_clients=attacks.shape[0]
        
    def __getitem__(self, index):
#         perm=torch.randperm(self.num_clients)
        data_out=self.data[index]
        label_out=self.label
        center_out=self.center[index]
        data_sorted=torch.sort(data_out)
        data_order=data_sorted[1]
        return data_out[data_order], [label_out[data_order],center_out]
#         return data_out,[label_out,center_out]
    def __len__(self):
        return self.data.shape[0]//10


# In[15]:


def get_loader(attack):
    label=torch.ones(10)
    for i in attacker_list_labelflipping[attack]:
        label[i]=0
    for i in attacker_list_omniscient[attack]:
        label[i]=0
    path=f'./AggData/{attack}/FedAvg_0.pt'
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


# train_loader, test_loader=get_loader(attacks[3])


# In[ ]:





# In[6]:


def train(net,train_loader,criterion,optimizer,device, on):
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
    accuracy_mean = 0
    accuracy_median = 0
    count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target[1].to(device)
            outputs = net(data)
            accuracy+=F.mse_loss(outputs[1], target)
            accuracy_mean+=F.mse_loss(data.mean(1).view(-1,1), target)
            accuracy_median+=F.mse_loss(data.median(1)[0].view(-1,1), target)
            count+=len(data)
    print('Accuracy: %.4E (%.4E, %.4E)' % (accuracy/count,accuracy_mean/count,accuracy_median/count ))
    return accuracy/count, accuracy_mean/count, accuracy_median/count


# In[7]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def dot_product(A,B):
    return torch.bmm(A.view(A.shape[0],1,A.shape[1]),B.view(B.shape[0],B.shape[1],1))
class Mlp(nn.Module):
    def __init__(self,n,in_dim):
        super(Mlp, self).__init__()
        self.in_dim=in_dim
        self.fc1 = nn.Linear(self.in_dim, n,bias=False)
        self.fc2 = nn.Linear(n, n,bias=False)

        self.fc3 = nn.Linear(n, self.in_dim,bias=False)

    def forward(self, input):
        x = input.view(-1, self.in_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.softmax(self.fc3(x),dim=1)
        pred=dot_product(x,input).squeeze(-1)
    
        return x,pred


# In[8]:


class CNN(nn.Module):
    def __init__(self,n,in_dim):
        super(CNN, self).__init__()
        self.in_dim=in_dim
        self.fc1 = nn.Conv1d(1, n, kernel_size=3,dilation=1, padding=1)
        self.fc2 = nn.Conv1d(n, n, kernel_size=3,dilation=2, padding=2)
        self.fc3 = nn.Conv1d(n, self.in_dim, kernel_size=3,dilation=1, padding=1)

        self.maxpool1=nn.AdaptiveMaxPool1d(1)
        
#         self.fc2 = nn.ConvTranspose1d(n, 10,kernel_size=3)
#         self.maxpool2=nn.AdaptiveMaxPool1d(1)

    def forward(self, input):
        x = input.view(-1, 1, self.in_dim)
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = nn.LeakyReLU()(self.fc3(x))

        x = self.maxpool1(x)
#         x = (self.fc2(x))
#         x = self.maxpool2(x)
        x=x.squeeze()
        x = F.softmax(x,dim=1)
        pred=dot_product(x,input).squeeze(-1)
#         pred=x1
        return x, pred


# In[9]:


# net_mlp=Mlp(10,10).cuda()
# net_mlp(batch[0])[0].shape


# In[10]:


# net_cnn=CNN(32,10).cuda()
# net_cnn(batch[0].cuda())[0].shape


# In[11]:


from tensorboardX import SummaryWriter
def write(name,scalar):
    writer=SummaryWriter(f'./agg_logs/{name}')
    writer.add_scalar('mse', scalar, 0)
    writer.close()


# In[12]:


accuracy={}
device='cuda'
mode_name=['classification','regression']
for attack in attacks:
    train_loader, test_loader=get_loader(attack)
    for criterion in [torch.nn.MSELoss(),torch.nn.L1Loss(),torch.nn.BCELoss()]:
        for mode in [0,1]:
            net_cnn=CNN(32,10).cuda()
            net_mlp=Mlp(10,10).cuda()

            for net in [net_mlp,net_cnn]:
                
                training_alias=f'{attack}/{net.__class__.__name__}/{mode_name[mode]}/{criterion.__class__.__name__}'
                if training_alias in accuracy:
                    continue
                
                print('Start training of %s'%training_alias)
                optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
                for epoch in range(1):
                    train(net,train_loader,criterion,optimizer,device,mode)
                    score=test(net,test_loader,device,'')
                write(training_alias,score[0])
                accuracy[training_alias]=score[0]
    accuracy[f'{attack}_mean']=score[1]
    accuracy[f'{attack}_median']=score[2]


# In[ ]:


# net=net_mlp
# # criterion=torch.nn.L1Loss()
# criterion=torch.nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
# device='cuda'
# for epoch in range(10):
#     train(net,train_loader,criterion,optimizer,device,1)
#     score=test(net,test_loader,device,'')
# accuracy[f'{attack}_mlp_regression_mse']=score[0]


# In[ ]:


# net=net_cnn
# criterion=torch.nn.L1Loss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

# device='cuda'
# for epoch in range(10):
#     train(net,train_loader,criterion,optimizer,device,1)
#     score=test(net,test_loader,device,'')


# In[ ]:


# torch.sum(net(batch[0])[0],1)


# In[ ]:


# batch=next(iter(test_loader))


# In[ ]:


# torch.mean(batch[0],1)


# In[ ]:


# batch[1][1].view(-1)


# In[ ]:


# net(batch[0].cuda())[1].view(-1)


# In[ ]:




