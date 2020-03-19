#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(1)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

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
        #center= mean of honest clients = sum/num
        label=attacks/torch.sum(attacks)
        self.center=torch.sum(data_tensors*label,1).view(-1,1)
        self.num_clients=attacks.shape[0]
        self.n=int(np.floor(self.data.shape[0]*sample_fraction))
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
    from os import listdir
    attacks=listdir('./AggData')
    for epoch in range(2):
        for attack in attacks:            
            path=f'./AggData/{attack}/FedAvg_{epoch}.pt'
            label=torch.load(f'./AggData/{attack}/label.pt')
            dataset=Agg_dataset(path,label)
            datasets.append(dataset)
    dataset=torch.utils.data.dataset.ConcatDataset(datasets)
    print(f"Loaded {len(dataset)} samples.")

    validation_split = .2
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
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


# In[5]:


def train(net,train_loader,criterion,optimizer,device, on=0):
    net.to(device)
    num_batches=len(train_loader)
    printProgressBar(0, num_batches, prefix = '  Training Progress:', suffix = 'Complete', length = 50)
    for idx, (data,target) in enumerate(train_loader):
        data = data.to(device)
        target = target[on].to(device)
        optimizer.zero_grad()   
        output = net(data)
        loss = criterion(output[on], target)
        loss.backward()
        optimizer.step()
        printProgressBar(idx+1, num_batches, prefix = '  Training Progress:', suffix = 'Complete', length = 50)
    
def test(net,test_loader,device,message_prefix):
    net.to(device)
    BCEloss = 0
    accuracy = 0
    accuracy_binary = 0
    accuracy_mean = 0
    accuracy_median = 0
    count = 0
    num_batches=len(test_loader)
    printProgressBar(0, num_batches, prefix = 'Validation Progress:', suffix = 'Complete', length = 50)
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            label = target[0].to(device)
            target = target[1].to(device)            
            outputs = net(data)
            BCEloss        +=nn.BCELoss(reduction='sum')(outputs[0],label)
            accuracy       +=F.l1_loss(outputs[1],                       target, reduction="sum")
            accuracy_binary+=F.l1_loss(outputs[2],                       target, reduction="sum")
            accuracy_mean  +=F.l1_loss(data.mean(-1).unsqueeze(-1),      target, reduction="sum")
            accuracy_median+=F.l1_loss(data.median(-1)[0].unsqueeze(-1), target, reduction="sum")
            count+=len(data)
            printProgressBar(idx+1, num_batches, prefix = 'Validation Progress:', suffix = 'Complete', length = 50)
    print('%s: \t%.4E \t %.4E \t%.4E \t%.4E (BCE: %.4E)' % (message_prefix,accuracy/count, accuracy_binary/count, accuracy_mean/count,accuracy_median/count,BCEloss/count ))
    return accuracy/count, accuracy_binary/count, accuracy_mean/count, accuracy_median/count, BCEloss/count


# In[11]:
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-vd","--vector_dimension",type=int,default=1,help="Dimension of weight feeding to the network")
    parser.add_argument("-l2","--weight_decay",type=float,default=0,help="Weight decay for l2 regularization")
    parser.add_argument("-n" ,"--model_name", type=str)
    parser.add_argument("-net","--network", type=str)
    
    args=parser.parse_args()
    torch.manual_seed(0)
    num_epoch=200
    batch_size=8192
    device='cuda'
    data_root='./data'
    vector_dimension=args.vector_dimension
    sample_fraction=0.05 #number of sample to draw in each experiments

    import allocateGPU
    allocateGPU.allocate_gpu()
    
    
    import baseline
    import aggNet
    import aggNet_Blocks
    import agg_Blocks_Multiple
    import rand_net
    networks={'baseline'   :baseline.Net,
              'aggNetRes'  :aggNet.Net,
              'aggNetBlock':aggNet_Blocks.Net,
              'aggNetBlockMultiple':agg_Blocks_Multiple.Net,
              'random':rand_net.Net
              
             }
    Net=networks[args.network]
    
    import torch.optim as optim





    from tensorboardX import SummaryWriter
    def write(name,tag, scalar,i=0):
        writer=SummaryWriter(f'./agg_logs/{name}')
        writer.add_scalar(tag, scalar, i)
        writer.close()


    # In[12]:


    accuracy_list={}


    # In[13]:


    train_loader, test_loader=get_concat_loader()


    for criterion in [torch.nn.BCELoss()]:
        mode=0
        for lr in [0.01]:

            net=Net(vector_dimension,10)


            training_alias=f'mulitple_attack/{args.model_name}/{criterion.__class__.__name__}/lr_{lr}'
            if training_alias in accuracy_list:
                continue

            print('Start training of %s'%training_alias)
            if not args.network=='random':
                optimizer = optim.Adam(net.parameters(), lr=lr , weight_decay=args.weight_decay)
            print('L1 loss:\tmodel \tmodel (binarized) \tmean \tmedian ')
            test(net,test_loader,device,f'Epoch -1')
            for epoch in range(num_epoch):
                if not args.network=='random':
                    train(net,train_loader,criterion,optimizer,device,mode)
                score=test(net,test_loader,device,f'Epoch {epoch}')
                if epoch%20==19:
                    torch.save(net.state_dict(),f'./aggNet/{args.model_name}_dim{vector_dimension}_{epoch}.pt')
                if epoch%10==9:
                    #use another set of data once in a while
                    train_loader, test_loader = get_concat_loader()
                write(training_alias+'/weighted', 'l1 loss', score[0],epoch)
                write(training_alias+'/binary', 'l1 loss',  score[1],epoch)  
                write(training_alias+'/mean', 'l1 loss',    score[2],epoch)
                write(training_alias+'/median', 'l1 loss',  score[3],epoch)
                write(training_alias+'/BCEloss', 'BCE loss',  score[4],epoch)



